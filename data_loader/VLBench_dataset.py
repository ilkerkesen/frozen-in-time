import json
import os
import os.path as osp
import random

import torch
import numpy as np
import pandas as pd
import decord

from base.base_dataset import TextVideoDataset, sample_frames


def _collate_fn(items):
    video = torch.cat([item['video'].unsqueeze(0) for item in items], dim=0)
    text, num_text = [], []
    for item in items:
        text.extend(item['text'])
        num_text.append(item['num_text'])
    key = [item['key'] for item in items]
    return {
        'video': video,
        'text': text,
        'num_text': num_text,
        'key': key,
    }


class VLBench(TextVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_metadata(self):
        json_path = os.path.join(self.metadata_dir, self.metadata_filename)
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        self.keys = list(self.metadata.keys())

    def _get_video_path(self, item):
        dataset = item['dataset']
        video_file = video_dir = video_path = None
        if dataset == 'QUVA':
            normalized = item.get('normalized')
            assert normalized
            video_dir = osp.join(self.quva_dir, 'normalized_videos')
            video_file = item['video_file']
        elif dataset == 'something-something-v2':
            video_dir = self.something_something_dir
            video_file = f'{item["dataset_idx"]}.webm'
        else:
            raise NotImplementedError('Not implemented yet.')
        video_path = os.path.join(video_dir, video_file)
        return video_path, video_file

    def _read_video(self, item, num_frames, sample='rand'):
        # raise NotImplementedError('implement this.')
        video_path, video_file = self._get_video_path(item)
        vr = decord.VideoReader(video_path, num_threads=1)
        start_time, end_time = item['start_time'], item['end_time']
        if item['time_unit'] == 'sec' and not (start_time == 0 and end_time == -1):
            raise NotImplementedError('this function has not been implemented for time unit of seconds.')
        if end_time is not None and end_time <= 0:
            end_time = None
        frames = vr[start_time:end_time]
        vlen = len(frames)
        frame_indices = sample_frames(num_frames, vlen, sample=sample)
        frames = frames[frame_indices]
        frames = frames.float() / 255.
        frames = frames.permute(0, 3, 1, 2)
        return frames, frame_indices

    def __getitem__(self, index):
        key = self.keys[index]
        item = self.metadata[key]
        caption = item['caption']
        foils = item['foils']
        if self.proficiency:
            caption = item['proficiency']['caption']
            foils = item['proficiency']['foils']
        frame_sample = 'uniform' if self.split == 'test' else 'rand'
        frames, frame_indices = self._read_video(
            item,
            self.video_params['num_frames'],
            sample=frame_sample,
        )

        if self.transforms is not None:
            frames = self.transforms(frames)

        final = torch.zeros([
            self.video_params['num_frames'],
            3,
            self.video_params['input_res'],
            self.video_params['input_res']
        ])
        final[:frames.shape[0]] = frames
        text = [caption] + foils
        data = {
            'video': final,
            'text': text,
            'key': key,
            'num_text': 1 + len(foils),
        }
        return data