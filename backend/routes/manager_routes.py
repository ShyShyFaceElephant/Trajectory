from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
from models import Member, Manager, LoginRequest, ManagerToken
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
from routes_extra.auth_utils import check_token_valid
import os
import jwt
import datetime
import shutil

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

### 取得醫生個人照 ###
@router.get("/manager/Profile/{manager_id}")
def get_manager_profile(manager_id: str):
    manager = manager_collection.find_one({"id": manager_id})
    if not manager or "profile_image_path" not in manager:
        raise HTTPException(status_code=404, detail="找不到該醫生或個人照")
    profile_path = manager["profile_image_path"]
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=404, detail="個人照不存在")
    return FileResponse(profile_path)

### 管理員註冊 ###
@router.post("/manager/Manager_Signup")
def manager_signup(
    id: str = Form(...),
    password: str = Form(...),
    department: str = Form(...),
    name: str = Form(...),
    profile_image_file: UploadFile = File(...)
):
    if manager_collection.find_one({"id": id}):
        raise HTTPException(status_code=400, detail="該醫生編號已註冊")

    profile_path = os.path.join(MANAGER_PROFILE_DIR, f"{id}.jpg")
    with open(profile_path, "wb") as f:
        shutil.copyfileobj(profile_image_file.file, f)

    new_manager = Manager(
        id=id,
        password=password,
        department=department,
        name=name,
        profile_image_path=profile_path,
        numMembers=0
    )
    manager_collection.insert_one(new_manager.dict())
    return {"message": "醫師註冊成功"}

### 管理員登入 ###
@router.post("/manager/Signin")
def manager_signin(signin_data: LoginRequest):
    manager = manager_collection.find_one({"id": signin_data.id})
    if not manager or manager["password"] != signin_data.password:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    
    manager_token = create_jwt_token({"id": signin_data.id, "role": "manager"})

    return {"manager_token": manager_token, "message": f"{signin_data.id} 成功登入"}

### 會員註冊 ###
@router.post("/manager/Member_Signup")
def member_signup(
    id: str = Form(...),
    sex: str = Form(...),
    name: str = Form(...),
    birthdate: str = Form(...),
    profile_image_file: UploadFile = File(...),
    managerToken: str = Form(...)
):
    decoded_token = check_token_valid(managerToken)
    if decoded_token["role"] != "manager":
        raise HTTPException(status_code=403, detail="僅限醫師新增會員")
    manager_id = decoded_token["id"]
    
    if member_collection.find_one({"id": id}):
        raise HTTPException(status_code=400, detail="該身份證字號已被註冊")

    profile_image_path = os.path.join(MEMBER_PROFILE_DIR, f"{id}.jpg")
    with open(profile_image_path, "wb") as buffer:
        shutil.copyfileobj(profile_image_file.file, buffer)

    new_member = Member(
        id=id,
        sex=sex,
        name=name,
        birthdate=birthdate,
        profile_image_path=profile_image_path,
        managerID=manager_id
    )

    new_member.generate_password()

    member_data = new_member.dict()
    member_collection.insert_one(member_data)

    manager_collection.update_one({"id": manager_id}, {"$inc": {"numMembers": 1}})

    return {"message": "成員註冊成功。"}

### 取得醫生資訊 ###
@router.post("/manager/Info")
def get_manager_info(manager_token: ManagerToken):
    manager_id = jwt.decode(manager_token.token, SECRET_KEY, algorithms=["HS256"])["id"]
    manager = manager_collection.find_one({"id": manager_id}, {"_id": 0, "password": 0})
    if not manager:
        raise HTTPException(status_code=404, detail="找不到該醫生")
    return manager

### 取得成員列表 ###
@router.post("/manager/MemberList")
def get_member_list(manager_token: ManagerToken):
    decoded_token = check_token_valid(manager_token.token)
    if decoded_token["role"] != "manager":
        raise HTTPException(status_code=403, detail="無權限")
    manager_id = decoded_token["id"]
    members = list(member_collection.find({"managerID": manager_id}, {"_id": 0, "password": 0}))
    return members