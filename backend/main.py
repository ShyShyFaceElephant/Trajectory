from fastapi import FastAPI
from routes.auth_routes import router as auth_router
from routes.manager_routes import router as manager_router
from routes.member_routes import router as member_router
from routes.ai_routes import router as ai_router

app = FastAPI()

app.include_router(auth_router)     # 沒有 prefix，是共用
app.include_router(manager_router)
app.include_router(member_router)
app.include_router(ai_router)

@app.get("/")
def home():
    return {"message": "API 正常運行中"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
