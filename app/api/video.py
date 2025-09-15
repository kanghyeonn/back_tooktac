from fastapi import APIRouter, WebSocket, WebSocketDisconnect  # 라우터/웹소켓 임포트
from sqlalchemy.orm import Session  # DB 세션 타입 힌트
from app.services.vision.posture_analyzer import PostureAnalyzer, PostureCoreModel, PostureSessionState  # 포즈 분석기 구성요소
from app.services.emotion.emotion_analyzer import EmotionAnalyzer, EmotionCoreModel, EmotionSessionState  # 감정 분석기 구성요소
from app.utils.auth_ws import get_user_id_from_websocket  # WebSocket에서 사용자 인증 정보 추출
from app.repository.analysis import VideoEvaluationResult  # 결과 저장용 ORM 모델
from app.repository.interview import InterviewQuestion, InterviewSession  # 세션/질문 ORM
from app.repository.database import SessionLocal  # DB 세션 팩토리
import numpy as np  # 바이트→배열 변환
import cv2  # 이미지 디코딩

router = APIRouter()  # FastAPI 라우터 생성

def get_db():
    # 의존성 주입 스타일의 DB 세션 생성기
    db = SessionLocal()  # 세션 팩토리 호출
    try:
        yield db  # 호출자에게 세션 제공
    finally:
        db.close()  # 사용 종료 시 닫기

# 무거운 모델은 프로세스당 1회 로드
GLOBAL_POSTURE_CORE = PostureCoreModel()  # MediaPipe 코어 로드
GLOBAL_EMOTION_CORE = EmotionCoreModel(model_path="/home/team3/tooktac/interview-evaluator/app/api/best.pt")  # YOLO 코어 로드

# 얇은 Analyzer는 코어를 참조만 함(상태 없음)
GLOBAL_POSTURE = PostureAnalyzer(GLOBAL_POSTURE_CORE)  # 포즈 분석기
GLOBAL_EMOTION = EmotionAnalyzer(GLOBAL_EMOTION_CORE)  # 감정 분석기

@router.websocket("/ws/expression")
async def expression_socket(websocket: WebSocket):
    await websocket.accept()  # 클라이언트 WebSocket 연결 수락

    analyzer = GLOBAL_POSTURE  # 전역 포즈 분석기 참조
    emotion_analyzer = GLOBAL_EMOTION  # 전역 감정 분석기 참조

    # 질문 생명주기 동안만 유지되는 상태 객체 생성
    posture_state = PostureSessionState()  # 포즈 누적 상태
    emotion_state = EmotionSessionState()  # 감정 누적 상태

    db: Session = next(get_db())  # DB 세션 획득
    try:
        user_id = await get_user_id_from_websocket(websocket)  # 쿠키/JWT 등에서 사용자 ID 추출

        # 쿼리 파라미터에서 question_order 추출
        order_str = websocket.query_params.get("question_id")  # question_id 파라미터 가져오기
        if not order_str or not order_str.isdigit():  # 유효성 검사
            await websocket.send_json({"error": "question_order가 유효하지 않습니다."})  # 에러 반환
            return  # 소켓 종료

        question_order = int(order_str)  # 정수 변환

        while True:
            data = await websocket.receive_bytes()  # 클라이언트가 전송한 바이너리 프레임 수신
            np_arr = np.frombuffer(data, np.uint8)  # 바이트를 NumPy 배열로 변환
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # JPEG/PNG 바이트를 BGR 이미지로 디코딩
            if frame is None:  # 디코딩 실패 검사
                await websocket.send_json({"expression": "이미지 변환 실패"})  # 에러 전송
                continue  # 다음 루프로 진행

            try:
                result = analyzer.analyze_frame(frame, posture_state)  # 포즈 분석 수행(상태 누적)
            except Exception as e:
                await websocket.send_json({"expression": "프레임 분석 실패"})  # 포즈 분석 에러 통지
                continue  # 다음 프레임으로 진행

            try:
                emotion_analyzer.analyze_frame(frame, emotion_state)  # 감정 분석 수행(상태 누적)
            except Exception:
                pass  # 감정 분석 실패는 무시하고 진행

            # 프론트에 즉시 피드백 전송
            await websocket.send_json({"expression": result})  # 프레임별 결과 전송

            # 프레임/버퍼 참조 해제로 GC 유도
            del frame, np_arr, data  # 메모리 회수에 도움

    except WebSocketDisconnect:
        # 연결 종료 시 이 질문의 최종 결과 계산
        final_video = posture_state.finalize()  # 포즈 최종 점수 계산
        emo_sum = emotion_state.summary()  # 감정 요약 계산

        # 사용자 최신 세션 조회
        session = (
            db.query(InterviewSession)
            .filter_by(user_id=user_id)
            .order_by(InterviewSession.started_at.desc())
            .first()
        )
        if not session:
            return  # 세션 없으면 저장 스킵

        # 해당 세션의 question_order에 해당하는 질문 조회
        question = (
            db.query(InterviewQuestion)
            .filter_by(session_id=session.id, question_order=question_order)
            .first()
        )
        if not question:
            return  # 질문 없으면 저장 스킵

        # ORM 객체 생성 후 저장
        video_result = VideoEvaluationResult(
            user_id=user_id,  # 사용자 ID
            session_id=session.id,  # 세션 ID
            question_id=question.id,  # 질문 ID
            question_order=question_order,  # 질문 순번
            gaze_score=final_video["gaze_rate_score"],  # 정면 응시 점수
            shoulder_warning=final_video["shoulder_posture_warning_count"],  # 어깨 경고 수
            hand_warning=final_video["hand_posture_warning_count"],  # 손 경고 수
            posture_score=final_video["shoulder_hand_score"],  # 어깨+손 합산 점수
            final_video_score=final_video["video_score"],  # 최종 비디오 점수
            positive_rate=emo_sum.get("긍정", 0),  # 긍정 비율
            neutral_rate=emo_sum.get("중립", 0),  # 중립 비율
            negative_rate=emo_sum.get("부정", 0),  # 부정 비율
            tense_rate=emo_sum.get("긴장", 0),  # 긴장 비율
            emotion_best=emo_sum.get("best"),  # 최빈 감정
            emotion_score=emo_sum.get("score")  # 감정 점수
        )
        db.add(video_result)  # DB 세션에 추가
        db.commit()  # 커밋으로 저장

    except Exception:
        # 예외 발생 시에도 현재까지 상태로 저장 시도
        final_video = posture_state.finalize()  # 포즈 최종 점수
        emo_sum = emotion_state.summary()  # 감정 요약

        session = (
            db.query(InterviewSession)
            .filter_by(user_id=user_id)
            .order_by(InterviewSession.started_at.desc())
            .first()
        )
        if session:
            question = (
                db.query(InterviewQuestion)
                .filter_by(session_id=session.id, question_order=question_order)
                .first()
            )
            if question:
                video_result = VideoEvaluationResult(
                    user_id=user_id,
                    session_id=session.id,
                    question_id=question.id,
                    question_order=question_order,
                    gaze_score=final_video["gaze_rate_score"],
                    shoulder_warning=final_video["shoulder_posture_warning_count"],
                    hand_warning=final_video["hand_posture_warning_count"],
                    posture_score=final_video["shoulder_hand_score"],
                    final_video_score=final_video["video_score"],
                    positive_rate=emo_sum.get("긍정", 0),
                    neutral_rate=emo_sum.get("중립", 0),
                    negative_rate=emo_sum.get("부정", 0),
                    tense_rate=emo_sum.get("긴장", 0),
                    emotion_best=emo_sum.get("best"),
                    emotion_score=emo_sum.get("score")
                )
                db.add(video_result)  # 기록 추가
                db.commit()  # 저장 커밋

        # 프론트에 오류 알림(가능하면 마지막으로 시도)
        try:
            await websocket.send_json({"expression": "분석 중 오류 발생"})
        except Exception:
            pass  # 소켓이 이미 닫힌 경우 무시

    finally:
        # DB 세션 정리
        try:
            db.close()  # 세션 닫기
        except:
            pass  # 예외 무시


