from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta, date
import os
import json
from typing import List, Dict

from app.repository.database import SessionLocal
from app.services.user.dependencies import get_current_user
from app.repository.user import User
from app.repository.interview import InterviewSession, InterviewQuestion
from app.repository.analysis import EvaluationResult, VideoEvaluationResult
from app.repository.report import (
    FinalReportSummary, ReportStrength, ReportImprovement,
    ReportAreaScore, ReportQuestionScore
)
from app.services.report.final_report_processor import FinalEvaluationGenerator
from app.services.report.interview_data_formatter import generate_interview_json_from_session

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------------------------
# 기존의 최종 리포트 생성 API는 동일 (필요시 스키마 매핑 유지)
# -------------------------------------------------------------------

def _to_number_score(maybe_json) -> int:
    """
    ReportAreaScore.score가 JSON(dict) 또는 숫자로 저장될 수 있어
    프론트에서 기대하는 '정수 점수'로 통일한다.
    우선순위: score > value > total > average > 자체가 숫자
    """
    if isinstance(maybe_json, (int, float)):
        return int(round(maybe_json))
    if isinstance(maybe_json, dict):
        for key in ("score", "value", "total", "average"):
            if key in maybe_json and isinstance(maybe_json[key], (int, float)):
                return int(round(maybe_json[key]))
    # 그 외 형식은 0으로 처리
    return 0

def _normalize_qtype(name: str) -> str:
    """
    질문유형 문자열을 프론트에서 쓰는 5키로 정규화.
    """
    n = (name or "").strip().lower()
    mapper = {
        "개념": "concept", "개념설명": "concept", "concept": "concept",
        "기술": "technical", "기술형": "technical", "technical": "technical",
        "상황": "situation", "상황형": "situation", "situation": "situation",
        "행동": "behavior", "행동형": "behavior", "behavior": "behavior",
        "꼬리": "followUp", "꼬리질문": "followUp", "followup": "followUp", "follow_up": "followUp"
    }
    # 가장 긴 키부터 매칭
    for k, v in mapper.items():
        if k in n:
            return v
    return "concept"  # 기본값

def _avg_or_zero(values: List[int]) -> int:
    return int(round(sum(values) / len(values))) if values else 0

@router.post("/report/final")
def generate_final_report(
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    # 1) 사용자 최신 세션
    session = (
        db.query(InterviewSession)
        .filter_by(user_id=user_id)
        .order_by(InterviewSession.started_at.desc())
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="면접 세션이 없습니다")

    parsed_data = generate_interview_json_from_session(db, session.id)

    generator = FinalEvaluationGenerator(gemini_api_key=os.getenv("GOOGLE_API_KEY"))
    report = generator.generate_final_report_from_json(parsed_data)

    # 기존 보고서 제거 후 갱신
    existing_summary = db.query(FinalReportSummary).filter_by(
        user_id=user_id, session_id=session.id
    ).first()
    if existing_summary:
        db.query(ReportStrength).filter_by(report_id=existing_summary.id).delete()
        db.query(ReportImprovement).filter_by(report_id=existing_summary.id).delete()
        db.query(ReportAreaScore).filter_by(report_id=existing_summary.id).delete()
        db.query(ReportQuestionScore).filter_by(report_id=existing_summary.id).delete()
        db.delete(existing_summary)
        db.flush()

    summary = FinalReportSummary(
        user_id=user_id,
        session_id=session.id,
        total_score=report["total_evaluation"]["total_score"],
        rank=report["total_evaluation"]["rank"],
        grade=report["total_evaluation"]["grade"],
        grade_message=report["total_evaluation"]["grade_message"],
        personalized_advice=report["ai_advice"]["personalized_message"]
    )
    db.add(summary)
    db.flush()

    for s in report["ai_advice"]["top_strengths"]:
        db.add(ReportStrength(
            report_id=summary.id,
            title=s.get("title", ""),
            description=s.get("description", ""),
            score=s.get("score", 0)
        ))

    for i in report["ai_advice"]["improvements"]:
        db.add(ReportImprovement(
            report_id=summary.id,
            priority=i.get("priority", 1),
            title=i.get("title", ""),
            description=i.get("description", ""),
            score=i.get("score", 0)
        ))

    for area_name, area_score in report["area_scores"].items():
        db.add(ReportAreaScore(
            report_id=summary.id,
            area_name=area_name,
            score=area_score
        ))

    for idx, q in enumerate(report["question_scores"]):
        summary_data = q.get("summary", "")
        summary_value = summary_data if isinstance(summary_data, (dict, list, str)) else str(summary_data)
        db.add(ReportQuestionScore(
            report_id=summary.id,
            question_order=idx + 1,
            question_name=q.get("name", ""),
            question_type=q.get("type", ""),
            question_text=q.get("question", ""),
            user_answer=q.get("my_answer", ""),
            model_answer=q.get("model_answer", ""),
            score=q.get("score", 0),
            summary=summary_value
        ))

    db.commit()

    return {
        "evaluationData": {
            "totalScore": report["total_evaluation"]["total_score"],
            "rank": report["total_evaluation"]["rank"],
            "grade": report["total_evaluation"]["grade"],
            "gradeMessage": report["total_evaluation"]["grade_message"],
            "areaScores": report["area_scores"],
            "questionScores": [
                {
                    "name": q.get("name", ""),
                    "score": q.get("score", 0),
                    "type": q.get("type", ""),
                    "question": q.get("question", ""),
                    "myAnswer": q.get("my_answer", ""),
                    "modelAnswer": q.get("model_answer", ""),
                    "summary": q.get("summary", "")
                }
                for q in report["question_scores"]
            ]
        },
        "aiAdvice": {
            "personalizedMessage": report["ai_advice"]["personalized_message"],
            "topStrengths": report["ai_advice"]["top_strengths"],
            "improvements": report["ai_advice"]["improvements"]
        }
    }

# ---------------------------------------------------
# 날짜별 상세 리포트: areas 숫자만 보장하도록 보정 (응답 포맷 유지)
# ---------------------------------------------------
@router.get("/report/date/{date}")
def get_report_by_date(
        date: str,
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    from datetime import datetime

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요.")

    session = (
        db.query(InterviewSession)
        .filter(
            InterviewSession.user_id == user_id,
            func.DATE(InterviewSession.started_at) == target_date
        )
        .order_by(InterviewSession.started_at.desc())
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="해당 날짜의 면접 기록이 없습니다.")

    summary = db.query(FinalReportSummary).filter_by(
        user_id=user_id, session_id=session.id
    ).first()
    if not summary:
        raise HTTPException(status_code=404, detail="해당 세션의 보고서를 찾을 수 없습니다.")

    strengths = db.query(ReportStrength).filter_by(report_id=summary.id).all()
    improvements = db.query(ReportImprovement).filter_by(report_id=summary.id).all()
    area_scores = db.query(ReportAreaScore).filter_by(report_id=summary.id).all()

    # area_name이 한국어/영어로 섞여 있을 수 있으니 매핑
    area_map = {
        "text": ["답변 내용", "텍스트", "text"],
        "voice": ["음성", "voice"],
        "video": ["영상", "video"],
        "emotion": ["감정", "emotion"],
    }

    def _find_area_score(key: str) -> int:
        names = set(a.lower() for a in area_map[key])
        for area in area_scores:
            if (area.area_name or "").lower() in names:
                return _to_number_score(area.score)
        return 0

    return {
        "success": True,
        "data": {
            "totalScore": summary.total_score,
            "rank": summary.rank,
            "areas": {
                "text": _find_area_score("text"),
                "voice": _find_area_score("voice"),
                "video": _find_area_score("video"),
                "emotion": _find_area_score("emotion"),
            },
            "topStrengths": [s.description for s in strengths],
            "improvements": [i.description for i in improvements],
            "aiAdvice": summary.personalized_advice
        }
    }

# ---------------------------------------------------
# 7일치 주간 데이터: areas + questionTypes까지 포함해 확장
# ---------------------------------------------------
@router.get("/training/weekly")
def get_weekly_training_data(
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    from datetime import datetime, timedelta

    today = datetime.now().date()
    week_ago = today - timedelta(days=6)
    weekly_data = []

    weekdays = ['월', '화', '수', '목', '금', '토', '일']

    for i in range(7):
        target_date = week_ago + timedelta(days=i)

    # 해당 날짜의 최신 세션 1건
        session = (
            db.query(InterviewSession)
            .filter(
                InterviewSession.user_id == user_id,
                func.DATE(InterviewSession.started_at) == target_date
            )
            .order_by(InterviewSession.started_at.desc())
            .first()
        )
        if not session:
            # 세션 없으면 스킵 (프론트는 없는 날짜는 라벨만, 데이터는 비움)
            continue

        summary = db.query(FinalReportSummary).filter_by(
            user_id=user_id, session_id=session.id
        ).first()
        if not summary:
            continue

        # 영역 점수
        area_scores = db.query(ReportAreaScore).filter_by(report_id=summary.id).all()
        def _area_score(key: str) -> int:
            mapping = {
                "text": ["답변 내용", "텍스트", "text"],
                "voice": ["음성", "voice"],
                "video": ["영상", "video"],
                "emotion": ["감정", "emotion"],
            }
            want = set(a.lower() for a in mapping[key])
            for area in area_scores:
                if (area.area_name or "").lower() in want:
                    return _to_number_score(area.score)
            return 0

        areas = {
            "text": _area_score("text"),
            "voice": _area_score("voice"),
            "video": _area_score("video"),
            "emotion": _area_score("emotion"),
        }

        # 질문유형별 평균 점수
        q_rows: List[ReportQuestionScore] = (
            db.query(ReportQuestionScore)
            .filter_by(report_id=summary.id)
            .all()
        )
        buckets: Dict[str, List[int]] = {
            "concept": [], "technical": [], "situation": [], "behavior": [], "followUp": []
        }

        for q in q_rows:
            score = int(q.score or 0)

            # 꼬리질문 조건: question_order == 3 또는 6
            if q.question_order in [3, 6]:
                buckets["followUp"].append(score)
            else:
                key = _normalize_qtype(q.question_type)
                buckets[key].append(score)

        # 평균 계산
        question_types = {k: _avg_or_zero(v) for k, v in buckets.items()}


        weekly_data.append({
            "date": target_date.strftime("%m/%d"),
            "fullDate": target_date.strftime("%Y-%m-%d"),
            "score": int(summary.total_score),
            "day": weekdays[target_date.weekday()],
            "areas": areas,
            "questionTypes": question_types,
            "routineAchieved": True  # 필요 시 스트릭 로직으로 교체
        })

    return {"success": True, "data": weekly_data}


@router.get("/training/day-counters")
def get_training_day_counters(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    # 1) 사용자의 모든 세션 중 첫 세션 날짜를 가져온다.
    first_session = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user_id)
        .order_by(InterviewSession.started_at.asc())
        .first()
    )

    # 2) 세션이 하나도 없으면 기본값 반환.
    if not first_session:
        return {
            "success": True,
            "data": {
                "programDayToday": 0,            # 프로그램 기준 n일차(없으면 0)
                "trainedDays": 0,                # 실제로 훈련한 날짜 수(중복 제외)
                "consecutiveStreak": 0,          # 오늘 포함 연속 스트릭
                "firstSessionDate": None,        # 첫 훈련일
                "dayIndexByDate": {}             # 날짜별 n일차 매핑
            }
        }

    # 3) 사용자 세션의 '날짜'만 distinct로 정렬해 가져온다.
    #    DATE(started_at)로 그룹핑해 중복 제거.
    distinct_dates: List[date] = [
        d[0] for d in (
            db.query(func.DATE(InterviewSession.started_at))
            .filter(InterviewSession.user_id == user_id)
            .group_by(func.DATE(InterviewSession.started_at))
            .order_by(func.DATE(InterviewSession.started_at).asc())
            .all()
        )
    ]

    # 4) 프로그램 기준 오늘 n일차 = (오늘 - 첫 세션일) + 1
    first_date = distinct_dates[0]
    today = datetime.now().date()
    program_day_today = (today - first_date).days + 1

    # 5) 실제 훈련한 '일수'(중복 제거된 날짜 수)
    trained_days = len(distinct_dates)

    # 6) 날짜 → n일차 매핑 만들기(첫 훈련일을 1일차로)
    day_index_by_date: Dict[str, int] = {
        d.strftime("%Y-%m-%d"): idx + 1
        for idx, d in enumerate(distinct_dates)
    }

    # 7) 연속 스트릭 계산:
    #    가장 최근 훈련일에서 하루씩 거꾸로 내려가며 연속성 확인.
    #    오늘에 훈련이 없으면 오늘 스트릭은 0이 아니고, '마지막 훈련일 기준 스트릭'을 반환.
    #    만약 오늘도 훈련했다면 today부터 카운팅됨.
    streak = 0
    if distinct_dates:
        last = distinct_dates[-1]
        streak = 1
        i = len(distinct_dates) - 2
        cur = last
        while i >= 0:
            if (cur - distinct_dates[i]) == timedelta(days=1):
                streak += 1
                cur = distinct_dates[i]
                i -= 1
            else:
                break

        # 선택: 오늘 날짜가 distinct_dates에 포함되지 않았다면,
        #       '오늘 연속'으로 표시하지 않도록 그대로 둔다.
        #       보통 UI에서는 "연속 n일"은 '마지막 기록 기준'으로 보여줘도 충분함.

    return {
        "success": True,
        "data": {
            "programDayToday": program_day_today,           # 오늘이 '몇일차'
            "trainedDays": trained_days,                    # 실제 훈련한 날짜 수
            "consecutiveStreak": streak,                    # 연속 스트릭
            "firstSessionDate": first_date.strftime("%Y-%m-%d"),
            "dayIndexByDate": day_index_by_date            # 날짜별 n일차 매핑
        }
    }

@router.get("/rank/current")
def get_rank_current(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    # 1) 내 최신 세션 찾기
    latest_session = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user_id)
        .order_by(InterviewSession.started_at.desc())
        .first()
    )
    if not latest_session:
        raise HTTPException(status_code=404, detail="사용자의 최신 세션이 없습니다.")

    # 2) 내 최신 세션 점수
    my_score = (
        db.query(FinalReportSummary.total_score)
        .filter(
            FinalReportSummary.user_id == user_id,
            FinalReportSummary.session_id == latest_session.id
        )
        .scalar()
    )
    if my_score is None:
        raise HTTPException(status_code=404, detail="최신 세션의 최종 보고서가 없습니다.")

    # 3) 전 사용자 '최신 세션 점수' 집합 만들기
    #    - 각 사용자별 가장 최근 started_at을 구한 뒤, 그 세션의 total_score만 모음
    last_started_subq = (
        db.query(
            InterviewSession.user_id.label("user_id"),
            func.max(InterviewSession.started_at).label("last_started_at")
        )
        .group_by(InterviewSession.user_id)
        .subquery()
    )

    latest_scores_subq = (
        db.query(
            FinalReportSummary.total_score.label("total_score")
        )
        .join(
            InterviewSession,
            and_(
                FinalReportSummary.session_id == InterviewSession.id,
                FinalReportSummary.user_id == InterviewSession.user_id,
            )
        )
        .join(
            last_started_subq,
            and_(
                InterviewSession.user_id == last_started_subq.c.user_id,
                InterviewSession.started_at == last_started_subq.c.last_started_at
            )
        )
        .subquery()
    )

    # 4) 분포 내 위치 계산
    population = db.query(func.count()).select_from(latest_scores_subq).scalar() or 0
    higher_or_equal = (
        db.query(func.count())
        .select_from(latest_scores_subq)
        .filter(latest_scores_subq.c.total_score >= my_score)
        .scalar() or 0
    )

    rank_percent = round((higher_or_equal / population) * 100, 2) if population > 0 else 0.0

    return {
        "basis": "current",
        "my_score": int(my_score),
        "rank_percent": rank_percent,  # 상위 X%
        "population": int(population),
        "higher_or_equal": int(higher_or_equal)
    }


@router.get("/rank/best")
def get_rank_best(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    # 1) 내 최고 점수
    my_best_score = (
        db.query(func.max(FinalReportSummary.total_score))
        .filter(FinalReportSummary.user_id == user_id)
        .scalar()
    )
    if my_best_score is None:
        raise HTTPException(status_code=404, detail="사용자의 최고 점수가 없습니다.")

    # 2) 전 사용자 '최고 점수' 집합 만들기
    best_scores_subq = (
        db.query(
            FinalReportSummary.user_id.label("user_id"),
            func.max(FinalReportSummary.total_score).label("best_score")
        )
        .group_by(FinalReportSummary.user_id)
        .subquery()
    )

    # 3) 분포 내 위치 계산
    population = db.query(func.count()).select_from(best_scores_subq).scalar() or 0
    higher_or_equal = (
        db.query(func.count())
        .select_from(best_scores_subq)
        .filter(best_scores_subq.c.best_score >= my_best_score)
        .scalar() or 0
    )

    rank_percent = round((higher_or_equal / population) * 100, 2) if population > 0 else 0.0

    return {
        "basis": "best",
        "my_score": int(my_best_score),
        "rank_percent": rank_percent,  # 상위 X%
        "population": int(population),
        "higher_or_equal": int(higher_or_equal)
    }