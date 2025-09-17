import torch
import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
import nibabel as nib
import sys

# 定義 sfcn 的基本塊
class BasicBlock(nn.Module): # Conv -> BN -> MaxPooling -> ReLU
    expansion = 1  # 通道擴展倍數

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,  padding=1)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool1(out)
        out = torch.relu(out)
        return out
# Define the full network architecture
class SFCN(nn.Module):
    def __init__(self, num_classes=1):
        super(SFCN, self).__init__()
        
        # Initial channel size
        self.in_planes = 1
        # Block 1 (Input channel: 1 -> Output channel: 32)
        self.layer1 = self._make_layer(BasicBlock, 32, num_blocks=1, stride=1)

        # Block 2 (Input channel: 32 -> Output channel: 64)
        self.layer2 = self._make_layer(BasicBlock, 64, num_blocks=1, stride=1) 

        # Block 3 (Input channel: 64 -> Output channel: 128)
        self.layer3 = self._make_layer(BasicBlock, 128, num_blocks=1, stride=1)

        # Block 4 (Input channel: 128 -> Output channel: 256)
        self.layer4 = self._make_layer(BasicBlock, 256, num_blocks=1, stride=1)

        # Block 5 (Input channel: 256 -> Output channel: 256)
        self.layer5 = self._make_layer(BasicBlock, 256, num_blocks=1, stride=1)

        # Stage 2 (Conv1*1 -> BN -> Relu)
        self.conv1 = nn.Conv3d(256, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(64)
        
        # Stage 3 (AvgPool -> Dropout -> Conv1*1)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv3d(64, 50, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
    def _make_layer(self, block, out_planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, out_planes, stride))
        self.in_planes = out_planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stage 1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # Stage 2
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        # Stage 3
        x = F.adaptive_avg_pool3d(x, output_size=1)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], 50)
        x = self.softmax(x)
        # 權重平均
        weights = torch.arange(1, 51, dtype=x.dtype, device=x.device).view(1, -1)
        x = torch.sum(x * weights, dim=1, keepdim=True)
        x = x * 2
        return x
    
def runModel(path):
    # 載入五個模型
    cnns = [None] * 5
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir,'models')
    pt_files = os.listdir(model_dir)
    if len(pt_files) < 5:
        print("模型數量不足，請確認 models 資料夾內有至少五個模型檔")
        sys.exit(1)
    for i in range(5):
        cnns[i] = SFCN(num_classes=1)
        pt_path = os.path.join(model_dir, pt_files[i])
        # 載入模型參數
        cnns[i].load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')))
        cnns[i].eval()

    # 讀取影像資料
    try:
        data = nib.load(path)
        image = data.get_fdata()
    except Exception as e:
        print("讀取影像失敗:", e)
        sys.exit(1)

    # 製作 torch tensor：轉換為 float 並做 [0,1] normalization
    image_tensor = torch.from_numpy(image).to(torch.uint8)
    inputs = image_tensor.to(torch.float32) / 255.0
    # 增加 batch 與 channel 維度
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print("輸入影像尺寸:", inputs.shape)

    # 模型集成評估
    ensemble_output = 0.0
    for model_i in range(5):
        with torch.no_grad():
            outputs = cnns[model_i](inputs)
        ensemble_output += outputs.item() * 0.2

    brainAge  = np.round(ensemble_output, 1)
    print(f"腦齡：{brainAge}")
    return brainAge

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("請提供要預測的 nii.gz 檔案路徑")
        sys.exit(1)
    nii_file_path = sys.argv[1]
    brain_age = runModel(nii_file_path)
    # 將預測結果輸出到標準輸出，方便其他程式捕捉
    print(brain_age)