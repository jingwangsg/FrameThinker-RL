import re
import random
import time
import requests
import numpy as np
import requests
import base64
import json
import decord
import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from time import sleep
from PIL import Image
from io import BytesIO
from math import ceil, floor
from PIL import Image
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents
import cv2


class ThinkWithVideo(ToolBase):
    name = "think_with_video"
    action_start = '<action>'
    action_end = '</action>'
    chat_template = """<|im_end|>
        <|im_start|>user
        {}<|im_end|>
        <|im_start|>assistant
        """

    def __init__(self, _name, _desc, _params, **kwargs):
        self.chatml_history = []
        self.multi_modal_data = None
        self.video_path = None
        self.fps = None
        self.total_frames = None
        self.vr = None
        self.vr_highres = None  # High-resolution VideoReader for zoom action

        self.num_frames_per_sample = None

        self.height = 0
        self.width = 0
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        if not self.video_path:
            return '', 0.0, True, {}
        action_block = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_block:
            return '', 0.0, True, {}
        action_block = action_block[-1]

        time_match = re.match(r'get frame number at time\s+(\S+)', action_block.strip())
        if time_match:
            time_str = time_match.group(1)
            try:
                minutes, seconds = map(int, time_str.split(':'))
                total_seconds = minutes * 60 + seconds
                frame_number = int(total_seconds * self.fps)
                message = f"Frame number at time {time_str} is: {frame_number}."
                all_user_msg = self.chat_template.format(message)
                return all_user_msg, 0.0, False, {}
            except (ValueError, IndexError):
                return '', 0.0, True, {}

        # Handle "zoom in frame FRAME_INDEX" action
        zoom_match = re.match(r'zoom in frame\s+(\d+)', action_block.strip())
        if zoom_match:
            frame_idx = int(zoom_match.group(1))
            # Validate frame index
            if frame_idx < 0 or frame_idx >= self.total_frames:
                return '', 0.0, True, {}

            try:
                # Initialize high-res VideoReader if not already done
                if self.vr_highres is None:
                    max_retries = 3
                    base_delay = 2
                    for attempt in range(max_retries):
                        try:
                            # Use original video resolution (no scaling)
                            self.vr_highres = decord.VideoReader(
                                self.video_path,
                                ctx=decord.cpu(0)
                            )
                            break
                        except Exception as e:
                            if "Resource temporarily unavailable" in str(e) and attempt < max_retries - 1:
                                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                                print(
                                    f"WARN: [Attempt {attempt + 1}/{max_retries}] Failed to open video for zoom. "
                                    f"Retrying in {wait_time:.2f} seconds...")
                                time.sleep(wait_time)
                            else:
                                print(
                                    f"ERROR: [Attempt {attempt + 1}/{max_retries}] Failed to open video for zoom.")
                                raise e

                # Extract single high-resolution frame
                frame_array = self.vr_highres[frame_idx].asnumpy()
                frame_image = Image.fromarray(frame_array)

                # Format response
                user_msg = f"frame {frame_idx}: <image>"
                all_user_msg = self.chat_template.format(user_msg)
                obs_dict = {
                    "prompt": all_user_msg,
                    "multi_modal_data": {
                        "image": [frame_image]
                    }
                }
                return obs_dict, 0.0, False, {}
            except Exception as e:
                print(f"[ERROR] Failed to zoom in on frame {frame_idx} from video '{self.video_path}': {e}")
                return '', 0.0, True, {}

        match = re.search(r"choose frames between (\d+) and (\d+)", action_block)
        if not match:
            return '', 0.0, True, {}
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        if start_frame > self.total_frames:
            return '', 0.0, True, {}
        if end_frame > self.total_frames:
            end_frame = self.total_frames

        if start_frame >= end_frame - self.num_frames_per_sample:
            return '', 0.0, True, {}

        user_msg, focused_images_data = self._process_zoom_request_no_desc(
            start_frame=start_frame,
            end_frame=end_frame
        )
        all_user_msg = self.chat_template.format(user_msg)
        if len(focused_images_data) == 0:
            return '', 0.0, True, {}
        obs_dict = {
            "prompt": all_user_msg,
            "multi_modal_data": {
                "image": focused_images_data
            }
        }
        return obs_dict, 0.0, False, {}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.video_path = kwargs.get('video_path')
        self.fps = kwargs.get('fps')
        self.total_frames = kwargs.get('total_frames')
        self.height = kwargs.get('height')
        self.width = kwargs.get('width')
        self.vr = None
        self.vr_highres = None  # Reset high-res VideoReader
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data

        # Set number of frames based on video duration
        if self.total_frames and self.fps:
            duration_seconds = self.total_frames / self.fps
            if duration_seconds > 300:  # Long video (> 5 minutes)
                self.num_frames_per_sample = 12
            else:  # Short video (<= 5 minutes)
                self.num_frames_per_sample = 8
        else:
            self.num_frames_per_sample = 8

        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

    def _process_zoom_request_no_desc(
            self,
            start_frame: int,
            end_frame: int
    ) -> Tuple[str, List[np.ndarray]]:
        sample_start = start_frame + 1
        sample_end = end_frame - 1
        try:
            if self.vr is None:
                max_retries = 3
                base_delay = 2
                for attempt in range(max_retries):
                    try:
                        target_w, target_h = self._calculate_target_dims()
                        self.vr = decord.VideoReader(
                            self.video_path,
                            ctx=decord.cpu(0),
                            width=target_w,
                            height=target_h
                        )
                        break
                    except Exception as e:
                        if "Resource temporarily unavailable" in str(e) and attempt < max_retries - 1:
                            wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                            print(
                                f"WARN: [Attempt {attempt + 1}/{max_retries}] Failed to open video due to resource issue. "
                                f"Retrying in {wait_time:.2f} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(
                                f"ERROR: [Attempt {attempt + 1}/{max_retries}] Unrecoverable error or max retries reached. Raising exception.")
                            raise e

            frame_indices = sorted(
                list(set(map(int, np.linspace(sample_start, sample_end, self.num_frames_per_sample)))))

            focused_frames_array = self.vr.get_batch(frame_indices).asnumpy()
            assert len(frame_indices) > 0, f"Generated empty frame_indices for interval {sample_start}-{sample_end}"
        except Exception as e:
            print(f"[ERROR] Failed to process video '{self.video_path}' between frames {start_frame}-{end_frame}.")
            print(
                f"[ERROR] Frame indices attempted: {frame_indices if 'frame_indices' in locals() else 'Not generated'}")
            print(f"[ERROR] The original exception was: {e}")
            return "", []

        prompt_parts = []
        for frame_idx in frame_indices:
            prompt_parts.append(f"frame {frame_idx}: <image>")

        focused_prompt_segment = "\n".join(prompt_parts)
        focused_images_data = [Image.fromarray(frame) for frame in focused_frames_array]
        assert len(focused_images_data) > 0, self.vr
        return focused_prompt_segment, focused_images_data

    def _calculate_target_dims(self) -> tuple[int, int]:
        """Calculate target dimensions with short side = 256px."""
        w, h = self.width, self.height
        target_short_side = 256

        # Find the shorter side
        short_side = min(w, h)

        # If already small enough, keep original
        if short_side <= target_short_side:
            return w, h

        # Scale so that short side = 256
        scale_ratio = target_short_side / short_side
        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)

        return new_w, new_h