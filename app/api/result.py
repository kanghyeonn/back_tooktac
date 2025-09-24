# app/api/routes/result.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.repository.database import SessionLocal
from app.models.analysis import EvaluationResult, VideoEvaluationResult
from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.services.user.dependencies import get_current_user
from app.services.score.scoring import QuestionTypeWeights

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/result/full/latest")
def get_full_latest_result(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    print("=== /result/full/latest ===")
    print("user_id:", user_id)
    # 1. 가장 최근 세션 가져오기
    latest_session = (
        db.query(InterviewSession)
        .filter_by(user_id=user_id)
        .order_by(InterviewSession.started_at.desc())
        .first()
    )
    if not latest_session:
        raise HTTPException(status_code=404, detail="latest_session 없음")
    print("latest_session:", latest_session)
    print("latest_session:", latest_session)

    # 2. 가장 마지막 질문 가져오기
    latest_question = (
        db.query(InterviewQuestion)
        .filter_by(session_id=latest_session.id)
        .order_by(InterviewQuestion.question_order.desc())
        .first()
    )
    if not latest_question:
        raise HTTPException(status_code=404, detail="latest_question 질문 없음")
    print("latest_question:", latest_question)

    # 3. 해당 질문의 답변 가져오기
    latest_answer = (
        db.query(InterviewAnswer)
        .filter_by(question_id=latest_question.id)
        .first()
    )
    # if not latest_answer:
    #     raise HTTPException(status_code=404, detail="latest_answer 없음")
    # 4. 텍스트/음성 평가 결과
    text_result = (
        db.query(EvaluationResult)
        .filter_by(question_id=latest_question.id)
        .order_by(EvaluationResult.created_at.desc())
        .first()
    )
    # if not text_result:
    #     raise HTTPException(status_code=404, detail="EvaluationResult 없음")

    # 5. 영상 평가 결과
    video_result = (
        db.query(VideoEvaluationResult)
        .filter_by(question_id=latest_question.id)
        .order_by(VideoEvaluationResult.created_at.desc())
        .first()
    )
    # if not video_result:
    #     raise HTTPException(status_code=404, detail="VideoEvaluationResult 없음")

    question_analysis = {
        "type": latest_question.question_type,
        "detailAnalysis": {
            "text": {"score": (text_result.final_text_score or 0) if text_result else 0},
            "voice": {"score": (text_result.final_speech_score or 0) if text_result else 0},
            "emotion": {"score": (video_result.emotion_score or 0) if video_result else 0},
            "video": {"score": (video_result.final_video_score or 0) if video_result else 0}
        }
    }

    weighted_score = QuestionTypeWeights.calculate_weighted_score(question_analysis)

    # return {
    #     "session_id": latest_session.id,
    #     "question_order": latest_question.question_order,
    #     "question": latest_question.question_text,
    #     "user_answer": latest_answer.answer_text if latest_answer else "",
    #     "model_answer": text_result.model_answer or "",
    #     "strengths": text_result.strengths.split("\n") if text_result.strengths else [],
    #     "improvements": text_result.improvements.split("\n") if text_result.improvements else [],
    #     "final_feedback": text_result.final_feedback,
    #     "labels": {
    #         "speed": text_result.speed_label,
    #         "fluency": text_result.fluency_label,
    #         "tone": text_result.tone_label
    #     },
    #     "video": {
    #         "gaze_score": video_result.gaze_score if video_result else None,
    #         "shoulder_warning": video_result.shoulder_warning if video_result else None,
    #         "hand_warning": video_result.hand_warning if video_result else None,
    #     },
    #     "best_emotion": video_result.emotion_best if video_result else None,
    #     "weighted_score": weighted_score
    # }

    return {
        "session_id": latest_session.id,
        "question_order": latest_question.question_order,
        "question": latest_question.question_text or "",
        "user_answer": latest_answer.answer_text if latest_answer and latest_answer.answer_text else "",
        "model_answer": (text_result.model_answer if text_result and text_result.model_answer else ""),
        "strengths": (text_result.strengths.split("\n") if text_result and text_result.strengths else []),
        "improvements": (text_result.improvements.split("\n") if text_result and text_result.improvements else []),
        "final_feedback": (text_result.final_feedback if text_result and text_result.final_feedback else ""),
        "labels": {
            "speed": (text_result.speed_label if text_result and text_result.speed_label else ""),
            "fluency": (text_result.fluency_label if text_result and text_result.fluency_label else ""),
            "tone": (text_result.tone_label if text_result and text_result.tone_label else "")
        },
        "video": {
            "gaze_score": (video_result.gaze_score if video_result and video_result.gaze_score is not None else 0),
            "shoulder_warning": (video_result.shoulder_warning if video_result and video_result.shoulder_warning is not None else 0),
            "hand_warning": (video_result.hand_warning if video_result and video_result.hand_warning is not None else 0),
        },
        "best_emotion": (video_result.emotion_best if video_result and video_result.emotion_best else ""),
        "weighted_score": weighted_score
    }
