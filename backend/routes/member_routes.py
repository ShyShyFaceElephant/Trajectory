from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from models import LoginRequest, ManagerToken, MemberToken
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Union
from routes_extra.auth_utils import check_token_valid
import os
import jwt
import datetime

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

### 取得會員個人照 ###
@router.get("/member/Profile/{member_id}")
def get_member_profile(member_id: str):
    member = member_collection.find_one({"id": member_id})
    if not member or "profile_image_path" not in member:
        raise HTTPException(status_code=404, detail="找不到該會員或個人照")
    profile_path = member["profile_image_path"]
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=404, detail="個人照不存在")
    return FileResponse(profile_path)

### 成員登入 ###
@router.post("/member/Signin")
def member_signin(signin_data: LoginRequest):
    member = member_collection.find_one({"id": signin_data.id})
    if not member or signin_data.password != member["birthdate"]:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    token = create_jwt_token({"id": signin_data.id, "role": "member"})
    return {"member_token": token, "message": f"{signin_data.id} 成功登入"}

### 取得成員拍攝紀錄 ###
@router.post("/member/RecordsList")
def get_member_records(token: Union[MemberToken, ManagerToken], member_id: str):
    decoded = check_token_valid(token.token)
    user_id = decoded["id"]
    user_role = decoded["role"]

    if user_role == "manager":
        records = member_collection.find_one({"id": member_id}, {"_id": 0, "password": 0, "managerID": 0})
    elif user_role == "member" and user_id == member_id:
        records = member_collection.find_one({"id": member_id}, {"_id": 0, "password": 0, "managerID": 0})
    else:
        raise HTTPException(status_code=403, detail="無權查看此成員的紀錄")

    if not records:
        raise HTTPException(status_code=404, detail="找不到該成員的紀錄")

    return sorted(records.get("RecordList", []), key=lambda x: x["date"])

### 獲取單一成員基本資料 ###
@router.post("/member/Info")
def get_member_info(token: Union[MemberToken, ManagerToken], member_id: str):
    decoded = check_token_valid(token.token)
    user_id = decoded["id"]
    user_role = decoded["role"]

    if user_role == "manager":
        member = member_collection.find_one({"id": member_id}, {"_id": 0, "password": 0, "managerID": 0})
    elif user_role == "member" and user_id == member_id:
        member = member_collection.find_one({"id": member_id}, {"_id": 0, "password": 0, "managerID": 0})
    else:
        raise HTTPException(status_code=403, detail="無權查看此成員資料")

    if not member:
        raise HTTPException(status_code=404, detail="找不到該會員")

    return member