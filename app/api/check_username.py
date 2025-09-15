# app/api/check_username

from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from app.repository.database import SessionLocal
from app.repository.user import User

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/check-username")
def check_username(username: str = Query(...), db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    return {"available": existing_user is None}
