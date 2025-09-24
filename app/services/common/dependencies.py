from app.repository.database import SessionLocal

# DB 세션 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()