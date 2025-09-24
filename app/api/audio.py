from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from app.models.interview import InterviewAnswer, InterviewQuestion, InterviewSession
from app.services.feedback.speechfeedback import SpeechFeedbackGenerator
from app.services.speech.speech_analyzer import SpeechAnalyzer
from app.services.stt.stt_service import STTService
from app.utils.auth_ws import get_user_id_from_websocket
import tempfile
import os
import asyncio
import subprocess
import threading
from dotenv import load_dotenv
from app.repository.database import SessionLocal
from app.services.text.orchestrator import EvaluationOrchestrator
from app.models.analysis import EvaluationResult

load_dotenv()

router = APIRouter()

# 전역 싱글턴 저장소와 락
_EVAL_ORCH_SINGLETON = None
_EVAL_ORCH_LOCK = threading.Lock()

def get_orchestrator_singleton():
    """
    EvaluationOrchestrator를 프로세스 내에서 1회만 생성하고 재사용한다.
    멀티 쓰레드 환경에서 안전하게 최초 1회만 생성되도록 Lock을 사용한다.
    """
    global _EVAL_ORCH_SINGLETON
    if _EVAL_ORCH_SINGLETON is None:
        with _EVAL_ORCH_LOCK:
            if _EVAL_ORCH_SINGLETON is None:
                # 필요 시 환경변수/키 전달 가능
                # 예: EvaluationOrchestrator(api_key=os.getenv("OPENAI_API_KEY"))
                _EVAL_ORCH_SINGLETON = EvaluationOrchestrator()
    return _EVAL_ORCH_SINGLETON


@router.websocket("/ws/transcript")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    db: Session | None = None
    webm_path = None
    wav_path = None

    try:
        # 1) 인증
        user_id = await get_user_id_from_websocket(websocket)

        # 2) 쿼리스트링의 question_id는 실제로 question_order 값
        #    e.g. /ws/transcript?question_id=3  -> 3번째 질문(질문 순서 3)
        qorder_str = websocket.query_params.get("question_id") or websocket.query_params.get("questionId")
        if not qorder_str or not qorder_str.isdigit():
            await websocket.send_json({"error": "question_id(=question_order)가 유효하지 않습니다."})
            return
        question_order = int(qorder_str)

        # 3) DB 세션
        db = SessionLocal()

        # 4) 사용자 최신 세션 조회
        session = (
            db.query(InterviewSession)
            .filter(InterviewSession.user_id == user_id)
            .order_by(InterviewSession.started_at.desc())
            .first()
        )
        if not session:
            await websocket.send_json({"error": "세션을 찾을 수 없습니다."})
            return

        # 5) 최신 세션 내에서 question_order로 질문 조회
        question = (
            db.query(InterviewQuestion)
            .filter(
                InterviewQuestion.session_id == session.id,
                InterviewQuestion.question_order == question_order
            )
            .first()
        )
        if not question:
            await websocket.send_json({"error": f"세션 {session.id}에서 question_order={question_order} 질문을 찾을 수 없습니다."})
            return

        # 6) 오디오 수신
        data = await websocket.receive_bytes()
        if not data:
            _save_minimal_result(db, user_id, session.id, question, reason="빈 오디오")
            await websocket.send_json({"transcript": "", "feedback": _empty_feedback("오디오 데이터가 없습니다.")})
            return

        # 7) 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            webm_path = f.name

        await asyncio.sleep(0.1)
        wav_path = webm_path.replace(".webm", ".wav")

        # 8) ffmpeg 변환
        cmd = ["ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path, "-y", "-loglevel", "error"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            _save_minimal_result(db, user_id, session.id, question, reason="ffmpeg 변환 실패")
            await websocket.send_json({"transcript": "", "feedback": _empty_feedback("ffmpeg 변환 실패")})
            return

        # 9) STT (Clova)
        clova = STTService(stt_type="clova")
        clova_text, clova_raw = clova.transcribe(wav_path)
        text_clean = (clova_text or "").strip()

        if text_clean == "":
            _save_answer(db, session.id, question, user_id, "")
            _save_minimal_result(db, user_id, session.id, question, reason="음성 인식 불가")
            await websocket.send_json({"transcript": "", "feedback": _empty_feedback("음성 인식이 되지 않았습니다.")})
            return

        # 10) 답변 저장 후 추가 STT(Vito) 및 음성 분석
        _save_answer(db, session.id, question, user_id, text_clean)

        vito = STTService(stt_type="vito")
        vito_text, _ = vito.transcribe(wav_path)

        analyzer = SpeechAnalyzer(clova_raw)
        speed = analyzer.speech_speed_calculate()
        pitch = analyzer.calculate_pitch_variation(wav_path)
        fillers = analyzer.find_filler_words(vito_text)

        sf = SpeechFeedbackGenerator(speed, pitch, fillers).generate_feedback()
        labels = sf.get("labels", {}) or {}
        score_detail = sf.get("score_detail", {}) or {}
        total_score = sf.get("total_score", 0) or 0

        # 11) 텍스트 평가 (LLM)
        try:
            # orchestrator = EvaluationOrchestrator()
            # ev = orchestrator.evaluate_answer(question.question_text, text_clean, question.question_type)
            # 변경:
            orchestrator = get_orchestrator_singleton()
            ev = orchestrator.evaluate_answer(question.question_text, text_clean, question.question_type)
        except Exception:
            _save_minimal_result(
                db, user_id, session.id, question,
                reason="LLM 평가 실패",
                speech_scores=score_detail, labels=labels, total_speech=total_score
            )
            await websocket.send_json({"transcript": text_clean, "feedback": sf})
            return

        # 12) 최종 EvaluationResult 저장
        er = EvaluationResult(
            user_id=user_id,
            session_id=session.id,
            question_id=question.id,
            question_order=question.question_order,
            similarity=ev.get("similarity", 0.0),
            intent_score=ev.get("intent_score", 0.0),
            knowledge_score=ev.get("knowledge_score", 0.0),
            final_text_score=ev.get("final_score", 0),
            model_answer=ev.get("model_answer", "") or "",
            strengths="\n".join(ev.get("feedback", {}).get("strengths", [])),
            improvements="\n".join(ev.get("feedback", {}).get("improvements", [])),
            final_feedback=ev.get("feedback", {}).get("final_feedback", "") or "",
            speed_score=score_detail.get("speed"),
            filler_score=score_detail.get("filler"),
            pitch_score=score_detail.get("pitch"),
            final_speech_score=total_score,
            speed_label=labels.get("speed"),
            fluency_label=labels.get("fluency"),
            tone_label=labels.get("tone")
        )
        db.add(er)
        db.commit()

        await websocket.send_json({"transcript": text_clean, "feedback": sf})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        if db is not None:
            db.rollback()
        try:
            await websocket.send_json({"error": f"internal_error: {type(e).__name__}"})
        except Exception:
            pass
    finally:
        if db is not None:
            db.close()
        for p in (webm_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def _save_answer(db: Session, session_id: int, question, user_id: int, text: str) -> None:
    ans = InterviewAnswer(
        session_id=session_id,
        question_id=question.id,
        question_order=question.question_order,
        user_id=user_id,
        answer_text=text
    )
    db.add(ans)
    db.commit()


def _save_minimal_result(
    db: Session,
    user_id: int,
    session_id: int,
    question,
    reason: str,
    speech_scores: dict | None = None,
    labels: dict | None = None,
    total_speech: int | None = None
) -> None:
    speech_scores = speech_scores or {}
    labels = labels or {}
    er = EvaluationResult(
        user_id=user_id,
        session_id=session_id,
        question_id=question.id,
        question_order=question.question_order,
        similarity=0.0,
        intent_score=0.0,
        knowledge_score=0.0,
        final_text_score=0,
        model_answer="",
        strengths=f"{reason} - 강점 파악 불가",
        improvements=f"{reason} - 개선점 파악 불가",
        final_feedback=f"{reason}로 인해 텍스트 평가가 수행되지 않았습니다.",
        speed_score=speech_scores.get("speed", 0),
        filler_score=speech_scores.get("filler", 0),
        pitch_score=speech_scores.get("pitch", 0),
        final_speech_score=total_speech if total_speech is not None else 0,
        speed_label=labels.get("speed", "없음"),
        fluency_label=labels.get("fluency", "없음"),
        tone_label=labels.get("tone", "없음"),
    )
    db.add(er)
    db.commit()


def _empty_feedback(msg: str) -> dict:
    return {
        "feedback": msg,
        "score_detail": {"speed": 0, "filler": 0, "pitch": 0},
        "total_score_normalized": 0.0
    }
