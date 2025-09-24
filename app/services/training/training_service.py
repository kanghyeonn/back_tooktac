# app/services/user/training_service.py
from sqlalchemy.orm import Session
from fastapi import HTTPException
from typing import List, Dict, Optional
from app.repository.training_repository import (
    get_day_index_avg_scores,
    get_peer_avg_scores_up_to_my_days,
    get_latest_total_scores,
    get_latest_total_scores_fallback,
    get_job_scores_by_user,
    get_user_desired_job,
    get_user_max_day_index,
)

# 일차별 평균 점수 계산 (모든 사용자)
def calculate_daily_average_scores(db: Session, days: int) -> List[float]:
    rows = get_day_index_avg_scores(db, days)
    if not rows:
        raise HTTPException(status_code=404, detail="일차별 평균을 계산할 데이터가 없습니다.")

    day_to_avg: Dict[int, float] = {int(r[0]): float(r[1]) for r in rows}
    result: List[float] = []
    for d in range(1, days + 1):
        if d in day_to_avg:
            result.append(round(day_to_avg[d], 2))
    return result

# 로그인 사용자 기준: 다른 사용자들의 일차별 평균 점수
def calculate_peer_average_scores_up_to_my_days(
    db: Session, user_id: int, min_population: int
) -> Dict[str, object]:
    result_data = get_peer_avg_scores_up_to_my_days(db, user_id)
    my_days: Optional[int] = result_data.get("my_days")
    rows = result_data.get("rows")

    if not my_days or my_days < 1:
        raise HTTPException(status_code=404, detail="사용자의 학습 일수가 없습니다. 최소 1일 이상 학습 후 다시 시도하세요.")

    if not rows:
        raise HTTPException(status_code=404, detail="다른 사용자들의 일차별 데이터가 없습니다.")

    day_to_avg: Dict[int, Optional[float]] = {d: None for d in range(1, int(my_days) + 1)}
    for d, n, avg in rows:
        if int(n) >= min_population:
            day_to_avg[int(d)] = float(avg)

    result: List[Optional[float]] = [
        round(day_to_avg[d], 2) if day_to_avg[d] is not None else None
        for d in range(1, int(my_days) + 1)
    ]

    return {
        "success": True,
        "my_days": int(my_days),
        "min_population": int(min_population),
        "data": result,
    }

# 최상위 퍼센타일 기준 컷오프 점수 계산
def calculate_top_percent_cutoff(db: Session, percent: float, use_fallback=False) -> Dict:
    if not (0 < percent < 100):
        raise HTTPException(status_code=400, detail="percent는 0과 100 사이여야 합니다.")

    scores = (
        get_latest_total_scores_fallback(db)
        if use_fallback
        else get_latest_total_scores(db)
    )
    scores.sort(reverse=True)

    if not scores:
        raise HTTPException(status_code=404, detail="점수 데이터가 없습니다.")

    index = max(0, int(len(scores) * (percent / 100)) - 1)
    cutoff = scores[index]
    return {"success": True, "cutoff": round(cutoff, 2), "total": len(scores)}

# 로그인 사용자의 직무 기준 다른 사용자들의 점수 목록
def calculate_job_stats(
    db: Session, user_id: int, cap_to_my_days: bool = True, min_peer_points: int = 3
) -> Dict:
    desired_job = get_user_desired_job(db, user_id)
    max_day = get_user_max_day_index(db, user_id)

    if not desired_job:
        raise HTTPException(status_code=404, detail="사용자의 desired_job이 없습니다.")
    if not max_day:
        raise HTTPException(status_code=404, detail="사용자의 학습 일차가 존재하지 않습니다.")

    raw_rows = get_job_scores_by_user(
        db, desired_job, user_id, max_day, cap=cap_to_my_days
    )

    data = [
        {"user_id": row[0], "day_index": row[1], "score": float(row[2])}
        for row in raw_rows
        if row[2] is not None
    ]

    active_user_ids = set([d["user_id"] for d in data])

    return {
        "success": True,
        "job": desired_job,
        "days": max_day,
        "users": len(active_user_ids),
        "data": data,
    }
