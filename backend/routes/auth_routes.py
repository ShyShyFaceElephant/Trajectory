from fastapi import APIRouter, Form, HTTPException
import jwt
from typing import Set
from dotenv import load_dotenv
import os
from routes_extra.auth_utils import token_blacklist, check_token_valid


load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

router = APIRouter(tags=["auth"])

@router.post("/Logout")
def logout(token: str = Form(...)):
    decoded = check_token_valid(token)
    if decoded.get("role") not in {"member", "manager"}:
        raise HTTPException(status_code=403, detail="不支援的角色")
    token_blacklist.add(token)
    return {"message": f"{decoded['role']} 已成功登出"}
