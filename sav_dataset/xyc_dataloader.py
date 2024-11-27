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
    def __init__(self, sav_dir: str):
        self.sav_dir = sav_dir
        self.data = []  # 存储图像路径、帧编号、视频编号、masklet 编号等
        self._prepare_dataset()
        self.transform = True

    def _prepare_dataset(self):
        # 首先检查当前目录是否包含视频文件
        video_files = [f for f in os.listdir(self.sav_dir) if f.endswith('.mp4')]
        if video_files:
            # 如果当前目录直接包含数据，则处理当前目录
            self._sub_prepare_dataset(self.sav_dir)
        else:
            # 否则，遍历子目录
            for sub_dir in os.listdir(self.sav_dir):
                sub_sav_dir = os.path.join(self.sav_dir, sub_dir)
                if os.path.isdir(sub_sav_dir):
                    #print(f"Processing {sub_sav_dir}")
                    self._sub_prepare_dataset(sub_sav_dir)

    def _sub_prepare_dataset(self, sub_sav_dir):
        sub_sav_dataset = SAVDataset(sav_dir=sub_sav_dir)
        video_ids = [f.split('.')[0] for f in os.listdir(sub_sav_dir) if f.endswith('.mp4')]

        for video_id in video_ids:
            frames, manual_annot, auto_annot = sub_sav_dataset.get_frames_and_annotations(video_id)
            if auto_annot is None:
                continue

            video_frame_count = auto_annot["video_frame_count"]
            H = auto_annot["video_height"]
            W = auto_annot["video_width"]
            # 获取采样率，假设每个 masklet 使用相同的采样率
            sample_rate = 4

            # 遍历每个 masklet
            for masklet_id in auto_annot["masklet_id"]:
                # 获取该 masklet 的第一次出现的帧
                first_appeared_frame = auto_annot["masklet_first_appeared_frame"][masklet_id]
                masklet_frames = auto_annot["masklet_frame_count"][masklet_id]

                for annotated_frame_id in range(first_appeared_frame, masklet_frames):
                    # 获取两帧，确保不会超出视频帧数范围
                    anno_frame_id1 = annotated_frame_id
                    anno_frame_id2 = anno_frame_id1 + 1
                    if sample_rate*anno_frame_id2 >= video_frame_count:
                        break


                    self.data.append({
                        "sub_dir": sub_sav_dir,
                        "video_id": video_id,
                        "anno_frame_id1": anno_frame_id1,
                        "anno_frame_id2": anno_frame_id2,
                        "masklet_id": masklet_id

                    })

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
        video.set(cv2.CAP_PROP_POS_FRAMES, item['anno_frame_id2']*4)
        ret2, frame2 = video.read()
        if not ret2:
            raise ValueError(f"无法读取帧 {item['anno_frame_id2']*4} 来自视频 {item['video_id']}")
        img2_size = frame2.shape[:2]

        video.release()

        annotated_frame_id = item['anno_frame_id1']
        anno_frame_id2 = item['anno_frame_id2']
        masklet_id = item['masklet_id']
        rle1 = auto_annot["masklet"][annotated_frame_id][masklet_id]
        rle2 = auto_annot["masklet"][anno_frame_id2][masklet_id]

        # 获取解码后的 masklet
        mask1 = self._decode_masklet(rle1)
        mask2 = self._decode_masklet(rle2)

        if self.transform:
            frame1 = cv2.resize(frame1, (512, 512))
            frame2 = cv2.resize(frame1, (512, 512))
            # 将布尔类型掩码转换为 uint8 类型
            mask1 = (mask1.astype(np.uint8)) * 255
            mask2 = (mask2.astype(np.uint8)) * 255

            # 调整掩码的大小
            mask1_resized = cv2.resize(mask1, (512, 512))
            mask2_resized = cv2.resize(mask2, (512, 512))

            # 重新将掩码转换为二值形式（阈值化）
            mask1 = (mask1_resized > 127).astype(np.uint8)
            mask2 = (mask2_resized > 127).astype(np.uint8)
        # 返回两帧图像和对应的mask
        return frame1, mask1, frame2, mask2


from torchvision import transforms

resize_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 强制调整所有图像为 224x224
])

def save_batch_with_masks(images, masks, save_dir):
    batch_size = images.shape[0]  # 获取 batch 的大小

    # 如果保存目录不存在，则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(batch_size):
        image = images[i]
        mask = masks[i]

        # 将图像和掩码从 Tensor 转换为 numpy，如果它们是张量
        if torch.is_tensor(image):
            image = image.cpu().numpy()  # 从 Tensor 转换为 numpy 数组
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()  # 从 Tensor 转换为 numpy 数组

        # 如果图像的范围在 [0, 1]，将其转换为 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # 将 mask 转换为红色掩码 (R, G, B)
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = [255, 0, 0]  # 掩码为红色 (R)

        # 将 mask 叠加到图像上 (带有透明度)
        overlay = 0.3 * image + 0.7 * colored_mask

        # 创建一个单独的图形窗口，用于保存
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay.astype(np.uint8))
        plt.axis('off')  # 关闭坐标轴

        # 保存图像
        save_path = os.path.join(save_dir, f"image_with_mask_{i}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图形窗口，避免占用内存


if __name__ == "__main__":

    # 实例化数据集
    dataset = SAVMaskletDataset(sav_dir="/ailab/user/mahaoxuan/data/SAM/test/sav_004")

    # 使用 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 测试 DataLoader 输出
    for batch in dataloader:
        img1, mask1, img2, mask2 = batch
        #print(mask1.cpu().numpy())
        #print(auto_annot_path[0])
        #print(aa)
        #print(mm)

        print("Image 1 Shape:", img1.shape)
        print("Mask 1 Shape:", mask1.shape)
        print("Image 2 Shape:", img2.shape)
        print("Mask 2 Shape:", mask2.shape)
        save_batch_with_masks(img1, mask1, save_dir="/ailab/user/mahaoxuan/AlphaCLIP/temp_vis_save")
        break

