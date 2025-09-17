from torchvision import datasets, models, transforms
import torch
import os
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet264
import os.path as osp
import torch.nn as nn
import nibabel as nib
import pandas as pd
import numpy as np
import xgboost as xgb
# 檢查是否有 GPU 並設定裝置
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


def FE(model_path):
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
    return new_model

image_path = r"C:\API_Brain\3\AD_Risk\70 years\AD 78 years\002_S_0619.nii"
model_path = r"C:\API_Brain\3\AD_Risk\DenseNet121_15epochs_accuracy0.83378_val0.58201.pth"

img = sitk.ReadImage(image_path)
img = sitk.GetArrayFromImage(img) 
img = img.astype(np.float32)

image = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
model = FE(model_path)
output = model(image)
print(output)
params = {'gender':1,'age':78,'mmse':22}#Female:0/Male:1
patient_info = np.array([params['gender'],params['age'],params['mmse']], dtype=np.float32)
xgboost_model_path = r"C:\Users\USER\Desktop\dementia project\xgboost_model.json"
xgboost_model = xgb.XGBClassifier()
xgboost_model.load_model(xgboost_model_path)
output = output.detach().cpu().numpy().flatten()
final_input = np.concatenate([output, patient_info]).reshape(1,-1)
xgb_output = xgboost_model.predict(final_input)
print(xgb_output)
