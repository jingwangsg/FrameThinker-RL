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

import argparse
import os
import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/video_reason")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source_train = "data/video_reason/Video-Holmes/train.parquet"
    data_source_test = "data/video_reason/Video-Holmes/test.parquet"
    dataset_train = datasets.load_dataset("parquet", data_files=data_source_train, split="train")
    dataset_test = datasets.load_dataset("parquet", data_files=data_source_test, split="train")

    def make_map_fn():
        def process_fn(example):
            data_source = example.pop("data_source")
            prompt = example.pop("prompt")
            images = example.pop("images")
            ability = example.pop("ability")
            env_name = example.pop("env_name")
            reward_model = example.pop("reward_model")
            ground_truth = example.pop("ground_truth")
            question_type = example.pop("question_type")
            video_path = example.pop("video_path")
            new_video_path = "/opt/tiger/FrameThinker-RL/data" + video_path.split("s3:")[1].strip()
            metadata = example.pop("metadata")
            extra_info = example.pop("extra_info")
            data = {
                'data_source': data_source, 
                'prompt': prompt, 
                'images':images, 
                'ability': ability, 
                'env_name': env_name, 
                'reward_model': reward_model, 
                'ground_truth': ground_truth, 
                'question_type': question_type, 
                'video_path': new_video_path, 
                'metadata': metadata, 
                'extra_info': extra_info
            }
            return data

        return process_fn

    train_dataset = dataset_train.map(function=make_map_fn(), with_indices=False)
    test_dataset = dataset_test.map(function=make_map_fn(), with_indices=False)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
