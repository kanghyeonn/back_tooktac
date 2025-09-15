# app/api/signup.py

from fastapi import APIRouter, UploadFile, Form, File, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from app.repository.database import SessionLocal
from app.services.user.signup_service import register_user_with_resume
from app.services.user.background_tasks import structure_resume_background
import tempfile
import shutil
from pathlib import Path

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/signup")
async def signup(
    background_tasks: BackgroundTasks,
    username: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    nickname: str = Form(...),
    email: str = Form(...),
    birthdate: str = Form(...),
    desiredJob: str = Form(...),
    resume: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    temp_resume_path = None

    # ✅ 이력서 임시 저장 (구조화용)
    if resume:
        suffix = Path(resume.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(resume.file, temp_file)
            temp_resume_path = temp_file.name

    # ✅ 사용자 + 이력서 내용 저장 (구조화는 안함)
    user = register_user_with_resume(
        db=db,
        username=username,
        password=password,
        name=name,
        nickname=nickname,
        email=email,
        birthdate=birthdate,
        desired_job=desiredJob,
        resume=resume,
        skip_structure=True  # 구조화는 나중에
    )

    # ✅ 백그라운드로 구조화 작업 실행
    if temp_resume_path:
        background_tasks.add_task(structure_resume_background, temp_resume_path, user.id)

    return {"message": f"{username}님 가입 완료!", "user_id": user.id}
