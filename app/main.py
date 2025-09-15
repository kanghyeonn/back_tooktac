from fastapi import FastAPI
from app.api import audio_router 
from app.api import video_router
from app.api import signup_router
from app.api import user_router
from app.api import interview_router
from app.api import result_router
from app.api import report_router
from app.api import training_page_router
from fastapi.middleware.cors import CORSMiddleware
import app.repository.model_registry

print(api_key)

app = FastAPI()

origins = [
    "http://localhost:3000",   # 개발 환경
    "http://127.0.0.1:3000",   # 개발 환경 (IP 직접접속)
    "https://tooktac.shop",    # 운영 환경
    "https://www.tooktac.shop",
    "http://tooktac.shop:18080",
    "http://www.tooktac.shop:18080",
    "https://tooktac.shop:18080",
    "https://www.tooktac.shop:18080"
]

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 개발 환경에선 Next.js 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket 라우터 포함
app.include_router(audio_router)
app.include_router(video_router)
app.include_router(signup_router)
app.include_router(user_router)
app.include_router(interview_router)
app.include_router(result_router)
app.include_router(report_router)
app.include_router(training_page_router)