from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
from models import Record
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
from BrainAge.BrainAge import runPreprocessing
from BrainAge.BrainAge import runBrainage
from AD_prediction.AD_prediction import runModel as runPreAD
from nii_to_2D.nii_to_2D import runSlice, runGradCAMSlice
from routes_extra.auth_utils import check_token_valid
import os
import jwt
import datetime
import shutil
from typing import Dict, List
import base64
from starlette.responses import JSONResponse

# 載入 .env 環境變數
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
MEMBER_COLLECTION = os.getenv("MEMBER_COLLECTION")
MANAGER_COLLECTION = os.getenv("MANAGER_COLLECTION")
SECRET_KEY = os.getenv("SECRET_KEY")
STORAGE_ROOT = os.getenv("STORAGE_ROOT","data")

# 連接 MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
member_collection = db[MEMBER_COLLECTION]
manager_collection = db[MANAGER_COLLECTION]

# 創建 FastAPI 路由
router = APIRouter()

# 創建 JWT Token
def create_jwt_token(data: dict, expires_delta: datetime.timedelta = datetime.timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

# 本地存儲路徑
MANAGER_PROFILE_DIR = os.path.join(STORAGE_ROOT, "manager_profile")
MEMBER_PROFILE_DIR = os.path.join(STORAGE_ROOT, "member_profile")
BRAIN_IMAGE_DIR = os.path.join(STORAGE_ROOT, "brain_image")
for path in [MANAGER_PROFILE_DIR, MEMBER_PROFILE_DIR, BRAIN_IMAGE_DIR]:
    os.makedirs(path, exist_ok=True)
Path(STORAGE_ROOT).mkdir(parents=True, exist_ok=True)

@router.get("/ai/slice/all/{member_id}/{record_count}")
def get_all_slice_images_base64(member_id: str, record_count: int) -> Dict[str, List[str]]:
    record_id = f"record{str(record_count).zfill(3)}"
    result = member_collection.find_one(
        {"id": member_id},
        {"_id": 0, "RecordList": {"$elemMatch": {"record_id": record_id}}}
    )

    if not result or "RecordList" not in result or len(result["RecordList"]) == 0:
        raise HTTPException(status_code=404, detail="找不到指定的紀錄")

    record = result["RecordList"][0]
    folder_path = record["folder_path"]

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="找不到紀錄的資料夾")

    slices = {'axial': [], 'coronal': [], 'sagittal': [],'gradCAM_axial': [], 'gradCAM_coronal': [], 'gradCAM_sagittal': []}

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.endswith(".png"):
                plane = os.path.basename(root)
                if plane in slices:
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                        slices[plane].append(b64)

    return JSONResponse(content=slices)

### 上傳拍攝記錄 + 前處理 ###
## bottom：建立紀錄 ##
@router.post("/ai/upload/Record")
def upload_record(
    managerToken: str = Form(...),
    member_id: str = Form(...),
    date: str = Form(...),
    MMSE_score: Optional[int] = Form(None),
    image_file: UploadFile = File(...)
):
    decoded_token = check_token_valid(managerToken)
    if decoded_token["role"] != "manager":
        raise HTTPException(status_code=403, detail="非醫師帳號")
    member = member_collection.find_one({"id": member_id})
    if not member:
        raise HTTPException(status_code=404, detail="找不到該成員")

    record_count = member.get("record_count", 0) + 1
    record_id = f"record{str(record_count).zfill(3)}"
    member_folder = os.path.join(BRAIN_IMAGE_DIR, f"{member_id}_{str(record_count).zfill(3)}")
    os.makedirs(member_folder, exist_ok=True)

    if not image_file.filename.lower().endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="請上傳 .nii.gz 檔案")
    original_path = os.path.join(member_folder, "original.nii.gz")
    with open(original_path, "wb") as f:
        shutil.copyfileobj(image_file.file, f)

    abs_path = os.path.abspath(original_path)
    try:
        preprocessing_path = runPreprocessing(abs_path)
        if not preprocessing_path or not preprocessing_path.endswith(".nii.gz"):
            raise ValueError("前處理回傳路徑錯誤")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"前處理失敗: {str(e)}")

    final_pp_path = os.path.join(member_folder, "preprocessing.nii.gz")
    shutil.move(preprocessing_path, final_pp_path)

    birthdate = member["birthdate"]
    record = Record(
        member_id=member_id,
        record_id=record_id,
        date=date,
        MMSE_score=MMSE_score,
        folder_path=member_folder
    )
    record.compute_actual_age(birthdate)
    member_collection.update_one(
        {"id": member_id},
        {
            "$push": {"RecordList": record.dict()},
            "$set": {"record_count": record_count}
        }
    )
    return {"message": "成功上傳紀錄並完成前處理"}

### AI 推論（模型 + 風險）###
## bottom：AI計算 ##
@router.post("/ai/{member_id}")
def ai_brain_age(
    member_id: str,
    record_count: int,
    manager_token: str = Form(...)
):
    decoded_token = check_token_valid(manager_token)
    if decoded_token["role"] != "manager":
        raise HTTPException(status_code=403, detail="非醫師帳號")

    record_id = f"record{str(record_count).zfill(3)}"
    result = member_collection.find_one(
        {"id": member_id},
        {"_id": 0, "RecordList": {"$elemMatch": {"record_id": record_id}}}
    )
    if not result or "RecordList" not in result or len(result["RecordList"]) == 0:
        raise HTTPException(status_code=404, detail="找不到指定的紀錄")
    record_data = result["RecordList"][0]
    base_dir = os.path.dirname(os.path.dirname(__file__))
    folder_path = os.path.join(base_dir, record_data["folder_path"])
    actual_age = record_data["actual_age"]
    MMSE_score = record_data.get("MMSE_score")
    
    OG_image_path = os.path.join(folder_path, "original.nii.gz")
    PP_image_path = os.path.join(folder_path, "preprocessing.nii.gz")
    cam_image_path = os.path.join(folder_path, "gradCAM.nii.gz")
    if not os.path.exists(OG_image_path) or not os.path.exists(PP_image_path):
        raise HTTPException(status_code=404, detail="找不到 MRI 檔案")

    sex = member_collection.find_one({"id": member_id}, {"_id": 0, "sex": 1}).get("sex")
    try:
        brain_age = runBrainage(PP_image_path, cam_image_path)
        risk_score = runPreAD(MMSE_score=MMSE_score, original_image_path=OG_image_path,
                              actual_age=actual_age, sex=sex) if MMSE_score else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推論失敗: {str(e)}")

    update_result = member_collection.update_one(
        {"id": member_id, "RecordList.record_id": record_id},
        {"$set": {"RecordList.$.brain_age": brain_age, "RecordList.$.risk_score": risk_score}}
    )

    if update_result.modified_count == 0:
        raise HTTPException(status_code=500, detail="結果寫入資料庫失敗")

    return {
        "message": "AI 預測完成",
        "record_id": record_id,
        "brainAge": brain_age,
        "riskScore": risk_score
    }

### 2D切片+儲存結果 ###
## bottom：儲存 ##
@router.post("/ai/restore/{member_id}")
def ai_brain_age(
    member_id: str,
    record_count: int,
    manager_token: str = Form(...)
):
    decoded_token = check_token_valid(manager_token)
    if decoded_token["role"] != "manager":
        raise HTTPException(status_code=403, detail="非醫師帳號")

    record_id = f"record{str(record_count).zfill(3)}"
    result = member_collection.find_one(
        {"id": member_id},
        {"_id": 0, "RecordList": {"$elemMatch": {"record_id": record_id}}}
    )

    if not result or "RecordList" not in result or len(result["RecordList"]) == 0:
        raise HTTPException(status_code=404, detail="找不到指定的紀錄")

    record_data = result["RecordList"][0]
    base_dir = os.path.dirname(os.path.dirname(__file__))
    folder_path = os.path.join(base_dir, record_data["folder_path"])
    PP_image_path = os.path.join(folder_path, "preprocessing.nii.gz")
    cam_image_path = os.path.join(folder_path, "gradCAM.nii.gz")

    if not os.path.exists(PP_image_path):
        raise HTTPException(status_code=404, detail=f"找不到前處理 MRI 檔案!!{PP_image_path}")

    try:
        output_dir = runSlice(PP_image_path, folder_path)
        runGradCAMSlice(PP_image_path, cam_image_path, folder_path)
        return {"outputDirection": output_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切片失敗: {str(e)}")