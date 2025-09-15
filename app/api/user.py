from fastapi import APIRouter, Form, Depends, HTTPException, Response, Query
from sqlalchemy.orm import Session
from app.repository.database import SessionLocal
from app.services.user.login_service import authenticate_user
from app.services.user.dependencies import get_current_user
from app.repository.user import User

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

    # ✅ 쿠키에 JWT 저장
    response.set_cookie(
        key="access_token",
        value=token_data["access_token"],
        httponly=True,
        secure=True,
        samesite='none',
        max_age=60 * 60 * 2, # 2시간
        path="/",
        domain ='.tooktac.shop'
    )
    return {"message": "로그인 성공"}

@router.post("/logout")
def logout(response: Response):
    response.set_cookie(
        key="access_token",
        value="",
        httponly=True,
        max_age=0,
        path="/",
        secure=True,      # HTTPS 환경이면 반드시 True
        samesite="none",    # CSRF 방어용
        domain='.tooktac.shop'
    )
    return {"message": "로그아웃 완료"}

@router.get("/me")
def get_me(user=Depends(get_current_user), db: Session = Depends(get_db)):
    user_obj = db.query(User).filter(User.id == user).first()
    return {
        "user_id": user_obj.id,
        "nickname": user_obj.nickname,
        "desired_job": user_obj.desired_job,
        "status": "authenticated"
    }

@router.get("/check-username")
def check_username(username: str = Query(...), db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    return {"available": existing_user is None}