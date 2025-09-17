from torchvision import datasets, models, transforms
import torch
import os
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import numpy as np
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet264
import os.path as osp
import torch.nn as nn
import nibabel as nib
import pandas as pd
# 檢查是否有 GPU 並設定裝置
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# 固定隨機種子，保證實驗結果可重現
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def FE(model_path,resize_image):
    #載入模型參數
    #model_path為模型參數的路徑
    #resize_image為前處理完的影像
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=3).to(device)
    weights = torch.load(model_path,weights_only=False)
    model.load_state_dict(weights)

    #將模型的輸出層拆除=================================================================================
    model = nn.Sequential(*list(model.children())[:-1])
    class selfmodel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(output_size=1),
                nn.Flatten(start_dim=1, end_dim=-1)
            )

        def forward(self, x):
            x = self.model(x)
            return x

    model1 = selfmodel()
    new_model = nn.Sequential(model, model1).to(device)
    new_model.eval()
    #將前處理的影像輸入=================================================================================
    resize_image = sitk.GetArrayFromImage(resize_image)
    resize_image = resize_image.astype(np.float32)
    tensor_data = torch.tensor(resize_image, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    #將輸入放入模型並輸出=================================================================================
    output = new_model(tensor_data)
    return output


    