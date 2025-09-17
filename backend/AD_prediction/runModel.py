from preprocess import n4, reg, skull, min_max_normalization, resample
from extractor import FE
import numpy as np
import os
import xgboost as xgb
import sys

def RiskScore(original_image_path, MMSE_score, actual_age, sex):
    gender = {"F": "Female", "M": "Male"}.get(sex.upper(), "Female")
    actual_age = float(actual_age)
    MMSE_score = float(MMSE_score)
    gender_num = {"Female": 0, "Male": 1}[gender]
    patient_info = np.array([gender_num, actual_age, MMSE_score], dtype=np.float32)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    fixed_path = os.path.join(base_dir, r"AD_Risk\mni_icbm152_t1_tal_nlin_sym_09a.nii")
    model_path = os.path.join(base_dir, r"AD_Risk\DenseNet121_15epochs_accuracy0.83378_val0.58201.pth")
    xgb_path = os.path.join(base_dir, r"AD_Risk\xgboost_model.json")

    img = n4(original_image_path)
    img = reg(img, fixed_path)
    img = skull(img)
    img = min_max_normalization(img)
    img = resample(img)
    feat = FE(model_path, img).detach().cpu().numpy().flatten()
    xgb_input = np.concatenate([feat, patient_info]).reshape(1, -1)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    pred = xgb_model.predict(xgb_input)[0]
    label = {0: "AD(阿茲海默症)", 1: "CN(正常認知)", 2: "MCI(輕度認知障礙)"}.get(pred, "未知")
    prediction = label
    return prediction
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("請提供要預測的 nii.gz 檔案路徑")
        sys.exit(1)
    original_image_path = sys.argv[1]
    MMSE_score = sys.argv[2]
    actual_age = sys.argv[3]
    sex = sys.argv[4]
    AD_prediction = RiskScore(original_image_path, MMSE_score, actual_age, sex)
    # 將預測結果輸出到標準輸出，方便其他程式捕捉
    print(AD_prediction)
