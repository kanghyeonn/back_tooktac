# app/models/analysis.py

from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, ForeignKey, Float, String, Text, TIMESTAMP
from app.models.base import Base
from datetime import datetime, timezone

class EvaluationResult(Base):
    __tablename__ = "evaluation_result"

    # 기본 키: 자동 증가 정수형
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 외래 키: 사용자 ID (User 테이블 참조)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)

    # 외래 키: 세션 ID (InterviewSession 테이블 참조)
    session_id = Column(Integer, ForeignKey("interview_session.id"), nullable=False)

    # 외래 키: 질문 ID (InterviewQuestion 테이블 참조)
    question_id = Column(Integer, ForeignKey("interview_question.id"), nullable=False)

    # ✅ 새로 추가한 컬럼: 질문 순서값 (1~6)
    question_order = Column(Integer, nullable=False)

    # 텍스트 평가 점수 (의도, 지식, 유사도)
    similarity = Column(Float)
    intent_score = Column(Float)
    knowledge_score = Column(Float)
    final_text_score = Column(Integer)

    # 모범답안 및 피드백 텍스트
    model_answer = Column(Text)
    strengths = Column(Text)
    improvements = Column(Text)
    final_feedback = Column(Text)

    # 음성 평가 점수 (속도, 간투어, 음조, 최종)
    speed_score = Column(Integer)
    filler_score = Column(Integer) # 유창성
    pitch_score = Column(Integer)
    final_speech_score = Column(Integer)

    # 음성 분석 라벨 (속도, 유창성, 어조)
    speed_label = Column(String(20))
    fluency_label = Column(String(20))
    tone_label = Column(String(20))

    # 생성 시각 (UTC 기준)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

    # 관계 설정
    user = relationship("User", back_populates="evaluation_results")
    session = relationship("InterviewSession", back_populates="evaluation_results")
    question = relationship("InterviewQuestion", back_populates="evaluation_result")

class VideoEvaluationResult(Base):
    __tablename__ = "video_evaluation_result"

    id = Column(Integer, primary_key=True, autoincrement=True)

    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("interview_session.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("interview_question.id"), nullable=False)

    question_order = Column(Integer, nullable=False)  # ✅ 질문 순서 (1~6)

    gaze_score = Column(Integer)
    shoulder_warning = Column(Integer)
    hand_warning = Column(Integer)
    posture_score = Column(Integer)
    final_video_score = Column(Integer)

    positive_rate = Column(Integer)  # 긍정 %
    neutral_rate = Column(Integer)  # 중립 %
    negative_rate = Column(Integer)  # 부정 %
    tense_rate = Column(Integer)  # 긴장 %
    emotion_best = Column(String(20))  # 가장 높은 감정
    emotion_score = Column(Integer)  # 감정 총점

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

    # 관계 설정
    user = relationship("User", back_populates="video_results")
    session = relationship("InterviewSession", back_populates="video_results")
    question = relationship("InterviewQuestion", back_populates="video_result", uselist=False)
