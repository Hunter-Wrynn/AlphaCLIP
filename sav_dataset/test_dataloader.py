import os
import json
import torch
import numpy as np
import pycocotools.mask as mask_util
from torch.utils.data import get_worker_info

import json
import cv2
import torch

import gc
import json
import cv2

from torch.utils.data import Dataset, DataLoader

from sav_dataset.utils.sav_utils import SAVDataset
import matplotlib.pyplot as plt





from torch.utils.data import get_worker_info
import json

import numpy as np
import pycocotools.mask as mask_util


class SAVMaskletDataset(Dataset):
    def __init__(self, json_dir: str, transform=True):
        self.json_dir = json_dir
        self.transform = transform
        self.data = []  # 存储数据
        self._load_preprocessed_data()


    def _load_preprocessed_data(self):
        # 列出 json_dir 目录下的所有 JSON 文件
        json_files = [os.path.join(self.json_dir, f) for f in os.listdir(self.json_dir) if f.endswith('.json')]
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.data.extend(data)  # 将所有数据合并到 self.data 中


    def __len__(self):
        return len(self.data)

    def _decode_masklet(self, rle):
        # 使用 pycocotools 解码 RLE 编码
        mask = mask_util.decode(rle) > 0
        return mask.astype(np.uint8) * 255  # 转换为 uint8 类型的二值图像

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']

        # 构建自动注释路径
        sub_dir = item['sub_dir']
        auto_annot_path = os.path.join(sub_dir, f"{video_id}_auto.json")

        if not os.path.exists(auto_annot_path):
            print(f"{auto_annot_path} doesn't exist.")
            auto_annot = None
        else:
            auto_annot = json.load(open(auto_annot_path))          

        video_path = os.path.join(item['sub_dir'], video_id + '.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        # 使用 OpenCV 读取视频
        video = cv2.VideoCapture(video_path)
        frame_index = item['anno_frame_id1'] * 4
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = video.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_index} from video {video_id}")
        
        video.release()
        


        # 解码掩码
        annotated_frame_id = item['anno_frame_id1']
        masklet_id = item['masklet_id']
        rle = auto_annot["masklet"][annotated_frame_id][masklet_id]
        mask = self._decode_masklet(rle)

        # 调整掩码大小以匹配帧的尺寸
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            # 对 frame 和 mask 进行转换（调整大小）
            frame = cv2.resize(frame, (512, 512))
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255

        # 确保返回 float32 类型的张量
        frame_tensor = torch.from_numpy(frame).float()  # 确保是 float32
        mask_tensor = torch.from_numpy(mask).float()    # 确保是 float32

        return frame_tensor, mask_tensor

    def __del__(self):
        # 释放任何视频资源（如果需要）
        pass