# app/models/report.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, JSON
from sqlalchemy.orm import relationship
from app.models.base import Base
from datetime import datetime, timezone


class FinalReportSummary(Base):
    __tablename__ = "final_report_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("interview_session.id"), nullable=False)

    total_score = Column(Integer)
    rank = Column("rank_value", String(20))
    grade = Column(String(10))
    grade_message = Column(Text)
    personalized_advice = Column(Text)

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

    # 관계 설정
    user = relationship("User", back_populates="final_reports")
    session = relationship("InterviewSession", back_populates="final_report")
    strengths = relationship("ReportStrength", back_populates="report", cascade="all, delete-orphan")
    improvements = relationship("ReportImprovement", back_populates="report", cascade="all, delete-orphan")
    area_scores = relationship("ReportAreaScore", back_populates="report", cascade="all, delete-orphan")
    question_scores = relationship("ReportQuestionScore", back_populates="report", cascade="all, delete-orphan")


class ReportStrength(Base):
    __tablename__ = "report_strength"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("final_report_summary.id"), nullable=False)

    title = Column(String(100))
    description = Column(Text)
    score = Column(Integer)

    report = relationship("FinalReportSummary", back_populates="strengths")


class ReportImprovement(Base):
    __tablename__ = "report_improvement"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("final_report_summary.id"), nullable=False)

    priority = Column(Integer)
    title = Column(String(100))
    description = Column(Text)
    score = Column(Integer)

    report = relationship("FinalReportSummary", back_populates="improvements")


class ReportAreaScore(Base):
    """영역별 점수를 저장하는 테이블"""
    __tablename__ = "report_area_score"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("final_report_summary.id"), nullable=False)
    area_name = Column(String(100), nullable=False)
    score = Column(JSON)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

    report = relationship("FinalReportSummary", back_populates="area_scores")


class ReportQuestionScore(Base):
    """질문별 점수와 상세 정보를 저장하는 테이블"""
    __tablename__ = "report_question_score"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("final_report_summary.id"), nullable=False)
    question_order = Column(Integer, nullable=False)
    question_name = Column(String(200), nullable=False)
    question_type = Column(String(50), nullable=False)
    question_text = Column(Text, nullable=False)
    user_answer = Column(Text, nullable=False)
    model_answer = Column(Text, nullable=False)
    score = Column(Integer, nullable=False)
    
    # Option 1: Use JSON column for complex data
    summary = Column(JSON)  # Changed from Text to JSON
    
    # Option 2: If you want to keep it as Text, handle serialization properly
    # summary = Column(Text)
    
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

    report = relationship("FinalReportSummary", back_populates="question_scores")