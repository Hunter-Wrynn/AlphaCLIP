import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from torchvision import transforms
import alpha_clip
from model.model import IM2TEXT
from sav_dataset.xyc_dataloader import SAVMaskletDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToPILImage, ToTensor, Normalize
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

# 定义对比损失函数和其他函数（与之前相同）

def contrastive_loss(image_features, text_features):
    # 计算相似度矩阵，不使用 softmax
    similarity = image_features @ text_features.T
    similarity = similarity / 0.07

    similarity_softmax = similarity.softmax(dim=-1)
    loss1 = -torch.mean(torch.log(similarity_softmax.diag()))

    # 计算每行的cross-entropy loss（图像到文本的损失）
    image_to_text_loss = nn.CrossEntropyLoss()(similarity, torch.arange(similarity.size(0)).cuda())

    # 计算每列的cross-entropy loss（文本到图像的损失）
    text_to_image_loss = nn.CrossEntropyLoss()(similarity.T, torch.arange(similarity.size(0)).cuda())

    # 总损失为两部分的平均值
    loss2 = (image_to_text_loss + text_to_image_loss) / 2

    # 输出损失信息
    print(f"loss1: {loss1.item()} loss2: {loss2.item() }")
    print(similarity.size(0))
    # 最终返回损失，并对损失做批次大小归一化
    return 0.05 *( 0.1*loss1 +  loss2 ) 

def train(alpha_model, img2text, images, masks, preprocess, optimizer, device):
    #print(f"Original images shape: {images.shape}")  # e.g., [batch_size, height, width, channels]

    alpha = process_batch(masks, device)

    #print(f"Processed masks shape: {alpha.shape}")
    images = preprocess_images(images, preprocess, device).half().to(device)

    with torch.no_grad():
        image_features = alpha_model.visual(images, alpha)  # alpha_model 已经在正确的设备上

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.float()
    text_features = img2text(image_features)
    loss = contrastive_loss(image_features, text_features)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def preprocess_images(images, preprocess, device):
    to_pil = ToPILImage()
    transform = preprocess
    #print(images.shape)
    pil_images = [to_pil(images[i].cpu().permute(2, 0, 1)) for i in range(images.size(0))]
    transformed_images = [transform(pil_image) for pil_image in pil_images]
    return torch.stack(transformed_images).to(device)

def process_batch(masks, device):
    batch_size = masks.shape[0]
    binary_masks = []
    for i in range(batch_size):
        mask = masks[i]
        binary_mask = mask.to(torch.float32)
        transformed_mask = mask_transform((binary_mask * 255).byte().unsqueeze(0).to(torch.float32)).half().to(device).unsqueeze(0)
        binary_masks.append(transformed_mask)
    binary_masks = torch.cat(binary_masks, dim=0)
    return binary_masks


mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])


def train_worker(rank, args):
    # 设置分布式环境
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=args.world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 加载模型
    alpha_model, preprocess = alpha_clip.load(
        "ViT-L/14",
        alpha_vision_ckpt_pth="clip_l14_grit1m_fultune_8xe.pth",
        device=device
    )

    for param in alpha_model.parameters():
        param.requires_grad = False

    img2text = IM2TEXT(embed_dim=768, middle_dim=512, output_dim=768, n_layer=4).to(device)

    # 使用 DDP 包装模型
    img2text = nn.parallel.DistributedDataParallel(img2text, device_ids=[rank])
    alpha_model = alpha_model.to(device)  # 因为 alpha_model 的参数被冻结，无需使用 DDP 包装

    optimizer = optim.Adam(img2text.parameters(), lr=1e-4)

    # 调整数据集和数据加载器
    dataset = SAVMaskletDataset(sav_dir="/data1/mhx/zscriDATA")

    # 使用 DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
    dataset,
    batch_size=1024,
    sampler=sampler,
    num_workers=4,  # 根据你的系统资源调整
    pin_memory=True,
    prefetch_factor=2
)

    print("dataloader done")

    # 训练循环
    for epoch in range(args.epochs):
        print(f"开始第 {epoch} 轮训练...")
        sampler.set_epoch(epoch)
        
        if rank == 0:
            # 仅在主进程上显示进度条
            data_loader = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)
        else:
            data_loader = dataloader
        
        for batch in data_loader:
            frame1, mask1, _, _ = batch
            frame1 = frame1.to(device)
            mask1 = mask1.to(device)
            loss = train(alpha_model, img2text, frame1, mask1, preprocess, optimizer, device)
            
            if rank == 0:
                data_loader.set_postfix({'loss': loss})

        
        # ...（保存模型等）

        if epoch == 10:
            # 仅在 rank 0 上保存模型
            if rank == 0:
                torch.save(img2text.state_dict(), f'img2text_epoch_{epoch}.pth')
                print(f"第 10 轮训练完成，模型已保存。")

    # 清理
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='使用多GPU的训练脚本。')
    parser.add_argument('--gpu_numbers', type=int, default=2, help='使用的GPU数量。')
    parser.add_argument('--epochs', type=int, default=1, help='训练的轮数。')
    args = parser.parse_args()
    args.world_size = args.gpu_numbers

    mp.spawn(train_worker, nprocs=args.gpu_numbers, args=(args,))

if __name__ == "__main__":
    main()
