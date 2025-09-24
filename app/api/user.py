# app/api/user.py

from fastapi import APIRouter, Form, File, UploadFile, Depends, HTTPException, Response, Query, BackgroundTasks
from sqlalchemy.orm import Session
from app.services.user.login_service import authenticate_user
from app.services.common.dependencies import get_db
from app.services.user.dependencies import get_current_user
from app.services.user.signup_service import register_user_with_resume
from app.services.user.background_tasks import structure_resume_background
from app.models.user import User
from pathlib import Path
import tempfile
import shutil

router = APIRouter()

# 로그인 엔드포인트
@router.post("/login")
def login(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    token_data = authenticate_user(db=db, username=username, password=password)
    if not token_data:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    # JWT를 쿠키에 저장 (HttpOnly, Secure, SameSite 설정 포함)
    response.set_cookie(
        key="access_token",
        value=token_data["access_token"],
        httponly=True,
        secure=True,
        samesite='none',
        max_age=60 * 60 * 2,  # 2시간
        path="/",
        domain='.tooktac.shop'
    )
    return {"message": "로그인 성공"}

# 로그아웃 엔드포인트 (쿠키 제거)
@router.post("/logout")
def logout(response: Response):
    response.set_cookie(
        key="access_token",
        value="",
        httponly=True,
        max_age=0,
        path="/",
        secure=True,
        samesite="none",
        domain='.tooktac.shop'
    )
    return {"message": "로그아웃 완료"}

# 내 정보 조회 (토큰 기반 인증)
@router.get("/me")
def get_me(user=Depends(get_current_user), db: Session = Depends(get_db)):
    user_obj = db.query(User).filter(User.id == user).first()
    return {
        "user_id": user_obj.id,
        "nickname": user_obj.nickname,
        "desired_job": user_obj.desired_job,
        "status": "authenticated"
    }

# 아이디 중복 확인
@router.get("/check-username")
def check_username(username: str = Query(...), db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    return {"available": existing_user is None}

# 회원가입 + 이력서 업로드
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

    # 이력서 임시 저장 (구조화용)
    if resume:
        suffix = Path(resume.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(resume.file, temp_file)
            temp_resume_path = temp_file.name

    # 사용자 정보 + 이력서 저장 (resume 내용만 저장, 구조화는 나중에 실행)
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
        skip_structure=True
    )

    # 구조화 작업은 백그라운드에서 실행
    if temp_resume_path:
        background_tasks.add_task(structure_resume_background, temp_resume_path, user.id)

    return {"message": f"{username}님 가입 완료!", "user_id": user.id}
