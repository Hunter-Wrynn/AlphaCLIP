import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import pycocotools.mask as mask_util
import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pycocotools.mask as mask_util
from sav_dataset.utils.sav_utils import SAVDataset
import numpy as np
import matplotlib.pyplot as plt

class SAVMaskletDataset(Dataset):
    def __init__(self, json_dir: str):
        self.json_dir = json_dir
        self.data = []  # 存储预处理后的数据
        self._load_preprocessed_data()
        self.transform = True

    def _load_preprocessed_data(self):
        # 列出 json_dir 目录下的所有 JSON 文件
        json_files = [os.path.join(self.json_dir, f) for f in os.listdir(self.json_dir) if f.endswith('.json')]

        for json_file in json_files:
            #print(json_file)
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.data.extend(data)  # 将所有数据合并到 self.data 中

    def __len__(self):
        return len(self.data)

    def _decode_masklet(self, rle, img_size=None):
        # 使用 pycocotools 解码 RLE 编码，并调整大小
        mask = mask_util.decode(rle)>0
        if img_size !=None:
            mask = cv2.resize(mask, (img_size[1], img_size[0]))# 调整到与原图相同大小
        return mask

    def __getitem__(self, idx):
        item = self.data[idx]

        # load the json
        sub_dir = item['sub_dir']
        video_id = item['video_id']
        auto_annot_path = os.path.join(sub_dir, video_id + "_auto.json")
        if not os.path.exists(auto_annot_path):
            print(f"{auto_annot_path} doesn't exist.")
            auto_annot = None
        else:
            auto_annot = json.load(open(auto_annot_path))

        # 加载视频帧
        video_path = os.path.join(sub_dir, item['video_id'] + '.mp4')
        video = cv2.VideoCapture(video_path)

        # 获取图片1
        video.set(cv2.CAP_PROP_POS_FRAMES, item['anno_frame_id1']*4)
        ret1, frame1 = video.read()
        if not ret1:
            raise ValueError(f"无法读取帧 {item['anno_frame_id1']*4} 来自视频 {item['video_id']}")
        img1_size = frame1.shape[:2]

        # 获取图片2
        '''
        video.set(cv2.CAP_PROP_POS_FRAMES, item['anno_frame_id2']*4)
        ret2, frame2 = video.read()
        if not ret2:
            raise ValueError(f"无法读取帧 {item['anno_frame_id2']*4} 来自视频 {item['video_id']}")
        img2_size = frame2.shape[:2]
        '''

        video.release()

        annotated_frame_id = item['anno_frame_id1']
        #anno_frame_id2 = item['anno_frame_id2']
        masklet_id = item['masklet_id']
        rle1 = auto_annot["masklet"][annotated_frame_id][masklet_id]
        #rle2 = auto_annot["masklet"][anno_frame_id2][masklet_id]

        # 获取解码后的 masklet
        mask1 = self._decode_masklet(rle1)
        #mask2 = self._decode_masklet(rle2)

        if self.transform:
            frame1 = cv2.resize(frame1, (512, 512))
            #frame2 = cv2.resize(frame1, (512, 512))
            # 将布尔类型掩码转换为 uint8 类型
            mask1 = (mask1.astype(np.uint8)) * 255
            #mask2 = (mask2.astype(np.uint8)) * 255

            # 调整掩码的大小
            mask1_resized = cv2.resize(mask1, (512, 512))
            #mask2_resized = cv2.resize(mask2, (512, 512))

            # 重新将掩码转换为二值形式（阈值化）
            mask1 = (mask1_resized > 127).astype(np.uint8)
            #mask2 = (mask2_resized > 127).astype(np.uint8)
        # 返回两帧图像和对应的mask
        return frame1, mask1
