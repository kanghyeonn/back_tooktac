# app/services/report/interview_data_formatter.py
from collections import defaultdict
from sqlalchemy.orm import Session
from app.repository.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.repository.analysis import EvaluationResult, VideoEvaluationResult
from app.services.score.scoring import QuestionTypeWeights
from sqlalchemy.inspection import inspect
import json

def calculate_final_score(text_result: EvaluationResult, video_result: VideoEvaluationResult, question_type: str) -> int:
    """
    질문 유형에 따라 text/voice/video/emotion 점수를 가중 평균으로 계산
    """
    return QuestionTypeWeights.calculate_weighted_score({
        "type": question_type,
        "detailAnalysis": {
            "text": {"score": text_result.final_text_score or 0},
            "voice": {"score": text_result.final_speech_score or 0},
            "video": {"score": video_result.final_video_score or 0},
            "emotion": {"score": video_result.emotion_score or 0}
        }
    })


# 2) SQLAlchemy 객체 → 컬럼 dict로 안전 변환
def sa_to_dict(obj):
    if obj is None:
        return None
    mapper = inspect(obj).mapper
    return {c.key: getattr(obj, c.key) for c in mapper.column_attrs}

# 3) 긴 문자열 자르기(로그 가독성)
def truncate_values(d: dict, maxlen: int = 120) -> dict:
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > maxlen:
            out[k] = v[:maxlen] + f"...(+{len(v) - maxlen} more)"
        else:
            out[k] = v
    return out


def _pick_representative_for_order(
    db: Session, candidates: list[InterviewQuestion]
) -> InterviewQuestion:
    """
    동일 question_order에 속한 여러 질문(candidates) 중 대표 1개를 고른다.
    우선순위:
      1) EvaluationResult와 VideoEvaluationResult가 모두 존재하는 질문
         (둘 다 있으면 최신 id 우선)
      2) 그 외에는 id가 가장 큰 것(가장 최근 생성) 우선
    """
    scored = []
    for q in candidates:
        # 각 후보의 결과 존재 여부 확인
        text_res = db.query(EvaluationResult).filter_by(question_id=q.id).first()
        video_res = db.query(VideoEvaluationResult).filter_by(question_id=q.id).first()
        both = 1 if (text_res is not None and video_res is not None) else 0
        scored.append((both, q.id, q))  # both(0/1), 최신성 proxy(id), 객체

    # 우선 both가 1인 것들 중에서 id가 큰 것, 없으면 그냥 id가 큰 것
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]

def generate_interview_json_from_session(db: Session, session_id: int) -> dict:
    """
    주어진 세션 ID에 대해 6개의 질문 평가 결과를 FinalEvaluationGenerator에서 사용할 JSON 형식으로 반환
    """
    session = db.query(InterviewSession).filter_by(id=session_id).first()
    if not session:
        raise ValueError("해당 session_id에 해당하는 면접 세션이 없습니다")

    user = session.user
    #questions = session.questions

    # print("=" * 20)
    # print("questions : ", questions)
    # print("=" * 20)

    qs = (
        db.query(InterviewQuestion)
        .filter(InterviewQuestion.session_id == session_id)
        .order_by(InterviewQuestion.question_order.asc(), InterviewQuestion.id.asc())
        .all()
    )
    by_order: dict[int, list[InterviewQuestion]] = defaultdict(list)
    for q in qs:
        by_order[q.question_order].append(q)

    # 2) 각 order별 대표 질문 1개만 선정
    unique_questions: list[InterviewQuestion] = []
    for order in sorted(by_order.keys()):
        reps = _pick_representative_for_order(db, by_order[order])
        unique_questions.append(reps)

    # if len(questions) != 6:
    #     raise ValueError("해당 세션에는 질문이 6개가 존재해야 합니다")

    question_analyses = []

    for q in sorted(unique_questions, key=lambda x: x.question_order):
        question_id = q.id
        order = q.question_order

        print("=" * 20)
        print("questionId : ", question_id)
        print("questionOrder : ", order)
        print("=" * 20)

        answer = db.query(InterviewAnswer).filter_by(question_id=question_id).first()
        text_result = db.query(EvaluationResult).filter_by(question_id=question_id).first()
        video_result = db.query(VideoEvaluationResult).filter_by(question_id=question_id).first()

        print("=" * 20)
        print("[DEBUG] Answer dict:")
        print(json.dumps(truncate_values(sa_to_dict(answer)), ensure_ascii=False, indent=2, default=str))
        print("[DEBUG] TextResult dict:")
        print(json.dumps(truncate_values(sa_to_dict(text_result)), ensure_ascii=False, indent=2, default=str))
        print("[DEBUG] VideoResult dict:")
        print(json.dumps(truncate_values(sa_to_dict(video_result)), ensure_ascii=False, indent=2, default=str))
        print("=" * 20)

        # if not all([answer, text_result, video_result]):
        #     raise ValueError(f"{order}번 질문의 분석 데이터가 부족합니다")

        question_data = {
            "question_id": str(question_id),
            "question_number": order,
            "question_type": q.question_type,
            "final_score": calculate_final_score(text_result, video_result, q.question_type),
            "question_text": q.question_text,
            "user_answer": answer.answer_text,
            "model_answer": text_result.model_answer,
            "detail_analysis": {
                "text": {
                    "score": text_result.final_text_score,
                    "similarity": text_result.similarity,
                    "accuracy": text_result.knowledge_score,
                    "understanding": text_result.intent_score
                },
                "voice": {
                    "score": text_result.final_speech_score,
                    "speed": {"score": round(text_result.speed_score * 2.5)},
                    "fluency": {"score": round(text_result.filler_score * 2.5)},
                    "tone": {"score": round(text_result.pitch_score * 5.0)},
                    "speed_label": text_result.speed_label,
                    "fluency_label": text_result.fluency_label,
                    "tone_label": text_result.tone_label
                },
                "video": {
                    "score": video_result.final_video_score,
                    "gaze_rate": {"percentage": video_result.gaze_score},
                    "shoulder_posture": {"score": 100 - video_result.shoulder_warning * 10},
                    "hand_posture": {"score": 100 - video_result.hand_warning * 10}
                },
                "emotion": {
                    "score": video_result.emotion_score,
                    "positive": video_result.positive_rate,
                    "neutral": video_result.neutral_rate,
                    "nervous": video_result.tense_rate,
                    "negative": video_result.negative_rate
                }
            },
            "feedback": text_result.final_feedback,
            "strengths": text_result.strengths.split("\n") if text_result.strengths else [],
            "improvements": text_result.improvements.split("\n") if text_result.improvements else []
        }


        # def _safe(v, default=0):
        #     return v if v is not None else default

        # question_data = {
        #     "question_id": str(question_id),
        #     "question_number": order,
        #     "question_type": q.question_type,
        #     "final_score": calculate_final_score(
        #         text_result or EvaluationResult(), 
        #         video_result or VideoEvaluationResult(), 
        #         q.question_type
        #     ),
        #     "question_text": q.question_text,
        #     "user_answer": getattr(answer, "answer_text", None) or "",
        #     "model_answer": getattr(text_result, "model_answer", None) or "",
        #     "detail_analysis": {
        #         "text": {
        #             "score": _safe(getattr(text_result, "final_text_score", None)),
        #             "similarity": _safe(getattr(text_result, "similarity", None)),
        #             "accuracy": _safe(getattr(text_result, "knowledge_score", None)),
        #             "understanding": _safe(getattr(text_result, "intent_score", None)),
        #         },
        #         "voice": {
        #             "score": _safe(getattr(text_result, "final_speech_score", None)),
        #             "speed": {"score": int(round(_safe(getattr(text_result, "speed_score", None)) * 2.5))},
        #             "fluency": {"score": int(round(_safe(getattr(text_result, "filler_score", None)) * 2.5))},
        #             "tone": {"score": int(round(_safe(getattr(text_result, "pitch_score", None)) * 5.0))},
        #             "speed_label": getattr(text_result, "speed_label", None),
        #             "fluency_label": getattr(text_result, "fluency_label", None),
        #             "tone_label": getattr(text_result, "tone_label", None),
        #         },
        #         "video": {
        #             "score": _safe(getattr(video_result, "final_video_score", None)),
        #             "gaze_rate": {"percentage": _safe(getattr(video_result, "gaze_score", None))},
        #             "shoulder_posture": {"score": 100 - _safe(getattr(video_result, "shoulder_warning", None)) * 10},
        #             "hand_posture": {"score": 100 - _safe(getattr(video_result, "hand_warning", None)) * 10},
        #         },
        #         "emotion": {
        #             "score": _safe(getattr(video_result, "emotion_score", None)),
        #             "positive": _safe(getattr(video_result, "positive_rate", None)),
        #             "neutral": _safe(getattr(video_result, "neutral_rate", None)),
        #             "nervous": _safe(getattr(video_result, "tense_rate", None)),
        #             "negative": _safe(getattr(video_result, "negative_rate", None)),
        #         },
        #     },
        #     "feedback": getattr(text_result, "final_feedback", None) or "",
        #     "strengths": (getattr(text_result, "strengths", None) or "").split("\n") if getattr(text_result, "strengths", None) else [],
        #     "improvements": (getattr(text_result, "improvements", None) or "").split("\n") if getattr(text_result, "improvements", None) else [],
        # }

        question_analyses.append(question_data)

    return {
        "user_info": {
            "user_id": str(user.id),
            "user_nickname": user.nickname,
            "interview_id": f"session_{session_id}",
            "interview_date": session.started_at.date().isoformat(),
            "interview_duration": 25  # 기본값 (옵션)
        },
        "question_analyses": question_analyses
    }
