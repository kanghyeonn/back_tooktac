# app/models/__init__.py 또는 모든 모델이 import된 곳에서 수동 연결
from sqlalchemy.orm import relationship
from app.repository.interview import InterviewSession
from app.repository.report import FinalReportSummary

InterviewSession.final_report = relationship("FinalReportSummary", back_populates="session", uselist=False)
FinalReportSummary.session = relationship("InterviewSession", back_populates="final_report", uselist=False)
