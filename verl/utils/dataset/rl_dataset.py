# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.templates import get_message_template
from verl.utils.dataset.vision_utils import extract_frames, compute_target_size

from PIL import Image


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(
            config.get("cache_dir", "~/.cache/verl/rlhf")
        )
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get(
            "filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)
        )
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = (
            self.data_files if not use_origin_parquet else self.original_data_files
        )
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(
                src=parquet_file, cache_dir=self.cache_dir
            )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)[
                "train"
            ]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(
                    tokenizer.apply_chat_template(
                        doc[prompt_key], add_generation_prompt=True
                    )
                )
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(
                use_origin_parquet=True
            )  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(
                r"old dataloader ckpt file is used, please train from scratch for better ckpt performance"
            )

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        message_template_name = self.config.get("message_template", "default")
        apply_message_template = get_message_template(message_template_name)
        messages = apply_message_template(messages, config=self.config, **example)

        # print(messages)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import (
                process_image,
                process_raw_image,
                process_video,
            )

            multi_modal_data = {}
            origin_multi_modal_data = {}

            images = None
            images_pil = None
            if self.image_key in row_dict:
                # is video and video_reading_kwargs is not None, then extract frames dynamically
                # for image, we only have image_path (or none)
                if getattr(self.config, "media_reading_kwargs", None) is not None:
                    media_reading_kwargs = self.config.media_reading_kwargs
                    assert (
                        media_reading_kwargs["sampling_mode"] == "uniform"
                    ), "Only uniform sampling mode is supported for now"

                    size = media_reading_kwargs.get("size", 360)

                    if "video_path" in row_dict:
                        num_frames = media_reading_kwargs["num_frames"]

                        video_path = row_dict["video_path"]
                        if (
                            hasattr(self.config, "media_dir")
                            and self.config.media_dir is not None
                        ):
                            video_path = os.path.join(self.config.media_dir, video_path)

                        images_pil, frame_indices = extract_frames(
                            video_path=video_path, num_frames=num_frames, size=size,
                        )
                    elif "image_path" in row_dict:
                        image_path = row_dict["image_path"]
                        if (
                            hasattr(self.config, "media_dir")
                            and self.config.media_dir is not None
                        ):
                            image_path = os.path.join(self.config.media_dir, image_path)
                        image_pil = Image.open(image_path).convert("RGB")
                        width, height = image_pil.size
                        target_size = compute_target_size(width=width, height=height, size=size)
                        if width != target_size[0] or height != target_size[1]:
                            image_pil = image_pil.resize(target_size)
                        images_pil = [image_pil.convert("RGB")]
                    else:
                        images_pil = row_dict.get(self.image_key)
            
                    
            assert images_pil is not None, "No images found in row_dict: {row_dict}"

            videos = None
            if self.video_key in row_dict:
                videos = [
                    process_video(video) for video in row_dict.pop(self.video_key)
                ]
                multi_modal_data["video"] = [video.numpy() for video in videos]


            origin_multi_modal_data["image"] = [process_raw_image(image) for image in images_pil]
            multi_modal_data["image"] = [process_image(image) for image in images_pil]

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["origin_multi_modal_data"] = origin_multi_modal_data
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

            messages = self._build_messages(row_dict)
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")



        else:
            messages = self._build_messages(row_dict)
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if (
            self.processor is not None
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [
                torch.cat((text_position_ids, vision_position_ids), dim=0)
            ]  # (1, 4, seq_length)
        elif (
            self.processor is not None
            and "Glm4vImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [
                torch.cat((text_position_ids, vision_position_ids), dim=0)
            ]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        if self.image_key in row_dict:
            row_dict.pop(self.image_key)
        
        if self.video_key in row_dict:
            row_dict.pop(self.video_key)

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        extra_info = row_dict.get("extra_info", {})
        if extra_info:
            row_dict["fps"] = extra_info.get("fps", "30")
            row_dict["video_path"] = extra_info.get("video_path", "")
            if self.config.get("media_dir", None) and row_dict["video_path"] != "":
                row_dict["video_path"] = os.path.join(
                    self.config.get("media_dir"), row_dict["video_path"]
                )

            row_dict["total_frames"] = extra_info.get("total_frames", 0)
            row_dict["height"] = extra_info.get("height", 0)
            row_dict["width"] = extra_info.get("width", 0)
        
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
