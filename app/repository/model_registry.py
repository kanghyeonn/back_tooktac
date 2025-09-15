# app/repository/model_registry.py

"""
모든 SQLAlchemy 모델을 한 번에 import 하여
문자열 기반 relationship()이 안전하게 해석되도록 보장하는 레지스트리 모듈.
이 모듈을 앱 시작 시점에 반드시 한 번 import 하십시오.
"""

# Declarative Base를 공유하는 모든 모델 모듈을 import 한다.
# 이 import들은 사이드이펙트(매퍼 등록)를 위해 존재한다.

from app.repository import user as _user_module               # noqa: F401
from app.repository import resume as _resume_module           # noqa: F401
from app.repository import interview as _interview_module     # noqa: F401
from app.repository import analysis as _analysis_module       # noqa: F401
from app.repository import report as _report_module           # noqa: F401
