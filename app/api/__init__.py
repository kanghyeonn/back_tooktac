from .audio import router as audio_router
from .video import router as video_router
from .user import router as user_router
from .interview import router as interview_router
from .result import router as result_router
from .report import router as report_router
from .training_page import router as training_page_router

__all__ = ["audio_router", "video_router", "user_router", "interview_router", "result_router", "report_router", "training_page_router"]
