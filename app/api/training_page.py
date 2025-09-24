# app/api/training_page.py

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.services.common.dependencies import get_db
from app.services.user.dependencies import get_current_user
from app.services.training.training_service import (
    calculate_daily_average_scores,
    calculate_peer_average_scores_up_to_my_days,
    calculate_top_percent_cutoff,
    calculate_job_stats,
)

router = APIRouter()

@router.get("/training/averages")
def get_daily_average_scores(
    days: int = Query(7, ge=1, le=30, description="몇 일차까지 평균을 구할지 (기본 7)"),
    db: Session = Depends(get_db),
):
    return calculate_daily_average_scores(db=db, days=days)


@router.get("/training/peer-averages")
def get_peer_average_scores_up_to_my_days(
    min_population: int = Query(5, ge=1, description="일차별 평균 산출 최소 표본(미만이면 해당 일차 제외)"),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user),
):
    return calculate_peer_average_scores_up_to_my_days(db=db, user_id=user_id, min_population=min_population)


@router.get("/rank/cutoff")
def get_top_percent_cutoff(
    percent: float = Query(12.0, ge=0.1, le=100.0, description="상위 퍼센트(%) 예: 12.0"),
    min_population: int = Query(10, ge=1, description="최소 인원 기준(너무 표본이 작을 때 방지)"),
    db: Session = Depends(get_db),
):
    return calculate_top_percent_cutoff(db=db, percent=percent, min_population=min_population)


@router.get("/rank/job-stats")
def get_job_stats(
    cap_to_my_days: bool = Query(True, description="내 학습 일수 이내 데이터만 활용"),
    min_peer_points: int = Query(2, ge=1, description="성장률 계산에 필요한 최소 일수(최소 2개 점)"),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user),
):
    return calculate_job_stats(
        db=db,
        user_id=user_id,
        cap_to_my_days=cap_to_my_days,
        min_peer_points=min_peer_points
    )
