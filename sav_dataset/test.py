import pycocotools.mask as mask_util
import numpy as np

# RLE 编码示例
rle = {
    'size': [848, 480],
    'counts': 'i\\Y4<Qj05K4L4M3M2N3O000010O00001O00010O00000O101O0000000000000O010000000O10O10000O01000O100O010O1O1O1N2O1O0O2O1N20O01O1001N101O1N100O100O10O01O2O00000O1O10O1O11N1O1O100O1O2N1O1O3L4HTXk5'
}

# 使用 pycocotools 解码 RLE
binary_mask = mask_util.decode(rle)>0

# 检查结果
print(binary_mask.shape)  # 输出掩码尺寸，例如 (848, 480)
print(np.unique(binary_mask))  # 检查掩码中的唯一值，一般为 [0, 1]
