# app/api/routes/interview.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.text.make_question import InterviewQuestionGenerator
from app.models.interview import InterviewSession, InterviewQuestion
from app.repository.database import SessionLocal
from app.services.user.dependencies import get_current_user
import os

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/start-interview")
def start_interview(db: Session = Depends(get_db), user_id=Depends(get_current_user)):
    print("질문 생성 시작")
    # 인터뷰 세션 생성
    session = InterviewSession(user_id=user_id)
    db.add(session)
    db.flush()  # session.id 사용 가능

    # 질문 생성기
    print("질문 생성 ")
    generator = InterviewQuestionGenerator(os.getenv("GEMINI_API_KEY"))
    parsed = generator.load_structured_from_db(db, user_id)
    q1 = generator.generate_conceptual_question(parsed)
    print("질문 생성 완료")
    print("-" * 50)
    print(q1)
    # DB 저장
    db.add(InterviewQuestion(
        session_id=session.id,
        question_order=1,
        question_text=q1["question"],
        question_type=q1["question_type"]
    ))
    db.commit()

    return {
        "session_id": session.id,
        "question": q1["question"]
    }

    # # DB 저장
    # db.add(InterviewQuestion(
    #     session_id=session.id,
    #     question_order=1,
    #     question_text="이전 질문에서 보여주신 백엔드 성능 문제 해결 과정과 스크래핑 스타일 유지 문제 해결 경험이, 지원자분께서 궁극적으로 추구하시는 '사용자에게 실질적인 가치를 제공하는 개발'이라는 목표와 어떻게 연결되는지 설명해주시겠습니까?",
    #     question_type="개념설명형"
    # ))
    # db.commit()
    #
    # return {
    #     "session_id": session.id,
    #     "question": "데이터베이스 시스템에서 '트랜잭션'의 개념과 ACID 속성이 데이터 무결성 및 일관성에 어떻게 기여하는지 설명해주세요."
    # }


@router.post("/generate-question/{order}")
def generate_next_question(order: int, db: Session = Depends(get_db), user_id=Depends(get_current_user)):

    # ✅ 최신 세션 조회
    session = db.query(InterviewSession).filter(
        InterviewSession.user_id == user_id
    ).order_by(InterviewSession.started_at.desc()).first()
    if not session:
        raise HTTPException(status_code=404, detail="세션 없음")

    generator = InterviewQuestionGenerator(os.getenv("GEMINI_API_KEY"))
    parsed = generator.load_structured_from_db(db, user_id)

    # ✅ 질문 생성
    if order == 2:
        q = generator.generate_technical_question(parsed)
        qt = "기술형"
        #qu = "프로젝트 중 대용량 데이터를 처리하고 머신러닝/딥러닝 모델을 활용하여 유의미한 결과를 도출했던 경험이 있다면, 데이터 수집부터 모델 적용 및 결과 활용까지의 파이프라인을 어떤 기술 스택으로 어떻게 구현하셨나요?"
    elif order == 3:
        q1 = db.query(InterviewQuestion).filter_by(session_id=session.id, question_order=1).first()
        q2 = db.query(InterviewQuestion).filter_by(session_id=session.id, question_order=2).first()
        a1 = q1.answer.answer_text if q1 and q1.answer else ""
        a2 = q2.answer.answer_text if q2 and q2.answer else ""
        q = generator.generate_followup_resume_question(parsed, q1.question_text, a1, q2.question_text, a2)
        qt = "개념설명형"
        #qu = "데이터베이스 트랜잭션의 ACID 속성과 대용량 데이터를 다루는 머신러닝/딥러닝 파이프라인 구축 경험을 종합하여, 데이터 수집 또는 전처리 단계에서 데이터의 무결성 및 일관성을 확보하기 위해 ACID 개념을 어떻게 구체적으로 적용할 수 있을까요?"
    elif order == 4:
        q = generator.generate_situational_question(parsed)
        qt = "상황형"
        #qu = "만약 데이터 기반 서비스 출시 직전 핵심 데이터 오류로 긴급 수정이 필요하고, 이로 인해 팀의 기존 역할 및 출시 일정이 불확실해졌다면 어떻게 대처하시겠습니까?"
    elif order == 5:
        q = generator.generate_behavioral_question(parsed)
        qt = "행동형"
        #qu = "팀 협업 프로젝트에서 팀 내 소통 문제 해결 및 역할 조율을 통해 성공적으로 마무리했던 경험을 구체적으로 말씀해주세요."
    elif order == 6:
        q4 = db.query(InterviewQuestion).filter_by(session_id=session.id, question_order=4).first()
        q5 = db.query(InterviewQuestion).filter_by(session_id=session.id, question_order=5).first()
        a4 = q4.answer.answer_text if q4 and q4.answer else ""
        a5 = q5.answer.answer_text if q5 and q5.answer else ""
        q = generator.generate_followup_coverletter_question(parsed, q4.question_text, a4, q5.question_text, a5)
        qt = "상황형"
        #qu = "앞선 답변에서 보여주신 문제 해결 능력과 팀 협업 경험을 바탕으로, 핵심 데이터 오류 발생 시 팀원들과의 소통 및 역할 조정을 통해 어떻게 불확실성을 해소하고 함께 최적의 해결책을 찾아나가겠습니까?"
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 질문 순서")

    # ✅ DB에 질문 저장
    db.add(InterviewQuestion(
        session_id=session.id,
        question_order=order,
        question_text=q["question"],
        question_type=q["question_type"]
    ))
    db.commit()

    return {
        "session_id": session.id,
        "question": q["question"]
    }

    # db.add(InterviewQuestion(
    #     session_id=session.id,
    #     question_order=order,
    #     question_text="이전 질문에서 보여주신 백엔드 성능 문제 해결 과정과 스크래핑 스타일 유지 문제 해결 경험이, 지원자분께서 궁극적으로 추구하시는 '사용자에게 실질적인 가치를 제공하는 개발'이라는 목표와 어떻게 연결되는지 설명해주시겠습니까?",
    #     question_type="개념설명형"
    # ))
    # db.commit()
    #
    # return {
    #     "session_id": session.id,
    #     "question": "이전 질문에서 보여주신 백엔드 성능 문제 해결 과정과 스크래핑 스타일 유지 문제 해결 경험이, 지원자분께서 궁극적으로 추구하시는 '사용자에게 실질적인 가치를 제공하는 개발'이라는 목표와 어떻게 연결되는지 설명해주시겠습니까?"
    # }