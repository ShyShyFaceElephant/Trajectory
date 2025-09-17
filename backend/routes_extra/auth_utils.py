import jwt
from fastapi import HTTPException
from typing import Set
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

# 黑名單（模組層級變數，共用）
token_blacklist: Set[str] = set()

def check_token_valid(token: str):
    if token in token_blacklist:
        raise HTTPException(status_code=401, detail="Token 已失效")
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已過期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="無效的 Token")
