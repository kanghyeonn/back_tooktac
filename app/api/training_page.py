# app/api/training_page.py
from typing import List, Dict, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session
from app.repository.database import SessionLocal
from app.models.interview import InterviewSession
from app.models.user import User
from app.models.report import FinalReportSummary  # final_report_summary ORM 모델
from app.services.user.dependencies import get_current_user # 프로젝트에서 사용 중인 인증 의존성


router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# 1) 모든 사용자: 일차별 평균 점수
# -----------------------------
@router.get("/training/averages")
def get_daily_average_scores(
    days: int = Query(7, ge=1, le=30, description="몇 일차까지 평균을 구할지 (기본 7)"),
    db: Session = Depends(get_db),
):
    """
    모든 사용자에 대해 '캘린더 날짜별 마지막 세션'만 남기고,
    사용자별 날짜 순서를 DENSE_RANK()로 1일차, 2일차...를 매긴 뒤,
    일차(day_index)별 final_report_summary.total_score 평균을 계산한다.
    MySQL 8+ (윈도우 함수) 전제.
    """

    # 날짜 단위 컬럼
    session_date = func.date(InterviewSession.started_at).label("session_date")

    # 같은 유저-같은 날짜 내 '마지막 세션' 선정을 위한 ROW_NUMBER()
    rn_in_day = func.row_number().over(
        partition_by=(InterviewSession.user_id, session_date),
        order_by=InterviewSession.started_at.desc()
    ).label("rn_in_day")

    # 사용자별 학습 날짜의 순번을 부여하는 DENSE_RANK() -> 1일차, 2일차...
    day_index = func.dense_rank().over(
        partition_by=InterviewSession.user_id,
        order_by=session_date.asc()
    ).label("day_index")

    # 윈도 컬럼을 포함한 세션 서브쿼리
    sess_subq = (
        select(
            InterviewSession.id.label("session_id"),
            InterviewSession.user_id.label("user_id"),
            session_date,
            rn_in_day,
            day_index
        ).subquery()
    )

    # 하루의 마지막 세션만 남긴 서브쿼리
    last_of_day_subq = (
        select(
            sess_subq.c.session_id,
            sess_subq.c.user_id,
            sess_subq.c.session_date,
            sess_subq.c.day_index
        )
        .where(sess_subq.c.rn_in_day == 1)
        .subquery()
    )

    # 마지막 세션과 final_report_summary 조인 → day_index별 total_score 수집
    joined = (
        select(
            last_of_day_subq.c.day_index,
            FinalReportSummary.total_score
        )
        .join(FinalReportSummary, FinalReportSummary.session_id == last_of_day_subq.c.session_id)
        .where(FinalReportSummary.total_score.isnot(None))
        .subquery()
    )

    # 일차별 평균
    avg_query = (
        select(
            joined.c.day_index,
            func.avg(joined.c.total_score).label("avg_score")
        )
        .where(joined.c.day_index <= days)
        .group_by(joined.c.day_index)
        .order_by(joined.c.day_index.asc())
    )

    rows = db.execute(avg_query).all()
    if not rows:
        raise HTTPException(status_code=404, detail="일차별 평균을 계산할 데이터가 없습니다.")

    # 결과: [일차1 평균, 일차2 평균, ...] 형태
    # 비어있는 일차를 0/None으로 채우려면 여기서 보정하면 됨.
    day_to_avg: Dict[int, float] = {int(r[0]): float(r[1]) for r in rows}
    result: List[float] = []
    for d in range(1, days + 1):
        if d in day_to_avg:
            result.append(round(day_to_avg[d], 2))
        else:
            # 정책 1) 비어있는 일차는 건너뜀 → 주석 처리
            # 정책 2) 0으로 채움 → 아래 주석 해제
            # result.append(0.0)
            # 정책 3) None으로 채움 → 프론트에서 보간
            # result.append(None)  # 타입 허용 시
            pass

    # 존재하는 일차만 반환하려면 아래 한 줄을 사용
    # result = [round(float(r[1]), 2) for r in rows]

    return {"success": True, "data": result}


# ---------------------------------------------------------
# 2) 로그인 사용자 기준: 다른 사용자들의 일차별 평균 (내 일수까지)
# ---------------------------------------------------------
@router.get("/training/peer-averages")
def get_peer_average_scores_up_to_my_days(
    min_population: int = Query(5, ge=1, description="일차별 평균 산출 최소 표본(미만이면 해당 일차 제외)"),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user),
):
    """
    로그인 사용자의 '학습 일수'(내 최대 day_index)를 구하고,
    동일한 일차에 대해 '다른 사용자들'만 대상으로 final_report_summary.total_score 평균을 산출한다.
    하루에 여러 세션이면 그 날의 마지막 세션만 반영한다.
    결과는 [일차1 평균, ... 일차N 평균] 형태. N=내 학습 일수.
    MySQL 8+ (윈도우 함수) 전제.
    """

    session_date = func.date(InterviewSession.started_at).label("session_date")

    rn_in_day = func.row_number().over(
        partition_by=(InterviewSession.user_id, session_date),
        order_by=InterviewSession.started_at.desc()
    ).label("rn_in_day")

    day_index = func.dense_rank().over(
        partition_by=InterviewSession.user_id,
        order_by=session_date.asc()
    ).label("day_index")

    sess_subq = (
        select(
            InterviewSession.id.label("session_id"),
            InterviewSession.user_id.label("user_id"),
            session_date,
            rn_in_day,
            day_index
        ).subquery()
    )

    last_of_day_subq = (
        select(
            sess_subq.c.session_id,
            sess_subq.c.user_id,
            sess_subq.c.session_date,
            sess_subq.c.day_index
        )
        .where(sess_subq.c.rn_in_day == 1)
        .subquery()
    )

    # 내 최대 day_index = 내 학습 일수
    my_day_query = (
        select(func.max(last_of_day_subq.c.day_index))
        .where(last_of_day_subq.c.user_id == user_id)
    )
    my_days: Optional[int] = db.execute(my_day_query).scalar()
    if not my_days or my_days < 1:
        raise HTTPException(status_code=404, detail="사용자의 학습 일수가 없습니다. 최소 1일 이상 학습 후 다시 시도하세요.")

    # 다른 사용자들 + 내 일수까지
    joined = (
        select(
            last_of_day_subq.c.day_index,
            FinalReportSummary.total_score
        )
        .join(FinalReportSummary, FinalReportSummary.session_id == last_of_day_subq.c.session_id)
        .where(
            FinalReportSummary.total_score.isnot(None),
            last_of_day_subq.c.user_id != user_id,
            last_of_day_subq.c.day_index <= my_days
        )
        .subquery()
    )

    # 일차별 표본 수와 평균
    agg = (
        select(
            joined.c.day_index,
            func.count(joined.c.total_score).label("n"),
            func.avg(joined.c.total_score).label("avg_score")
        )
        .group_by(joined.c.day_index)
        .order_by(joined.c.day_index.asc())
    )
    rows = db.execute(agg).all()
    if not rows:
        raise HTTPException(status_code=404, detail="다른 사용자들의 일차별 데이터가 없습니다.")

    # 1..my_days까지 배열을 만들고, 표본 부족 일차는 None으로 둔다.
    day_to_avg: Dict[int, Optional[float]] = {d: None for d in range(1, int(my_days) + 1)}
    for d, n, avg in rows:
        if int(n) >= min_population:
            day_to_avg[int(d)] = float(avg)

    result: List[Optional[float]] = [
        round(day_to_avg[d], 2) if day_to_avg[d] is not None else None
        for d in range(1, int(my_days) + 1)
    ]

    return {"success": True, "my_days": int(my_days), "min_population": int(min_population), "data": result}


# ----------------------------------------------------------
# 3) 상위 N% 컷 점수 (모든 사용자 최신 세션의 total_score 대상)
#    - MySQL 8+ 우선, 미지원 시 대안 쿼리 경로 사용
# ----------------------------------------------------------
@router.get("/rank/cutoff")
def get_top_percent_cutoff(
    percent: float = Query(12.0, ge=0.1, le=100.0, description="상위 퍼센트(%) 예: 12.0"),
    min_population: int = Query(10, ge=1, description="최소 인원 기준(너무 표본이 작을 때 방지)"),
    db: Session = Depends(get_db),
):
    """
    모든 사용자별 '마지막 인터뷰 세션'의 total_score만 모아 상위 percent% 컷 점수를 반환.
    MySQL 8+면 윈도우 함수(ROW_NUMBER) 사용, 아니면 그룹조인 대안 수행.
    """

    # 시도 1) 윈도우 함수 (MySQL 8+)
    try:
        rn = func.row_number().over(
            partition_by=InterviewSession.user_id,
            order_by=InterviewSession.started_at.desc()
        ).label("rn")

        latest_sess_subq = (
            select(
                InterviewSession.id.label("session_id"),
                InterviewSession.user_id.label("user_id"),
                InterviewSession.started_at,
                rn
            ).subquery()
        )

        q = (
            select(FinalReportSummary.total_score)
            .select_from(FinalReportSummary)
            .join(latest_sess_subq, latest_sess_subq.c.session_id == FinalReportSummary.session_id)
            .where(
                latest_sess_subq.c.rn == 1,
                FinalReportSummary.total_score.isnot(None),
            )
        )
        scores = [row[0] for row in db.execute(q).all()]

    except Exception:
        # 시도 2) 윈도우 미지원: 사용자별 최신 started_at을 구해 다시 세션 조인
        latest_started_subq = (
            select(
                InterviewSession.user_id,
                func.max(InterviewSession.started_at).label("max_started_at")
            )
            .group_by(InterviewSession.user_id)
            .subquery()
        )

        latest_session_subq = (
            select(InterviewSession.id.label("session_id"), InterviewSession.user_id)
            .join(
                latest_started_subq,
                and_(
                    InterviewSession.user_id == latest_started_subq.c.user_id,
                    InterviewSession.started_at == latest_started_subq.c.max_started_at,
                ),
            )
            .subquery()
        )

        q = (
            select(FinalReportSummary.total_score)
            .select_from(FinalReportSummary)
            .join(latest_session_subq, latest_session_subq.c.session_id == FinalReportSummary.session_id)
            .where(FinalReportSummary.total_score.isnot(None))
        )
        scores = [row[0] for row in db.execute(q).all()]

    # 유효성 검사
    scores = [int(s) for s in scores if s is not None]
    population = len(scores)
    if population < min_population:
        raise HTTPException(
            status_code=400,
            detail=f"표본 크기 부족: 현재 {population}명. min_population={min_population} 이상일 때만 컷 산출이 가능합니다.",
        )
    if population == 0:
        raise HTTPException(status_code=404, detail="최신 세션의 total_score 데이터가 존재하지 않습니다.")

    # 상위 percent% 컷 계산
    # 내림차순 정렬 후 k = ceil(N * (percent/100)) 번째(1-indexed) 점수
    from math import ceil

    scores.sort(reverse=True)
    k = max(1, ceil(population * (percent / 100.0)))
    cutoff_score = scores[k - 1]

    num_higher_or_equal = sum(1 for s in scores if s >= cutoff_score)
    rank_percent_exact = (num_higher_or_equal / population) * 100.0

    return {
        "basis": "latest_session_per_user",
        "percent": percent,
        "cutoff_score": cutoff_score,
        "population": population,
        "higher_or_equal": num_higher_or_equal,
        "rank_percent_exact": round(rank_percent_exact, 2),
        "note": "내림차순 정렬 후 ceil(N*percent/100)번째 점수를 임계값으로 사용. 동점은 포함.",
    }

@router.get("/rank/job-stats")
def get_job_stats(
    # 내 타임라인 기준으로 동료 집단을 자르기 위한 옵션들
    cap_to_my_days: bool = Query(True, description="내 학습 일수 이내 데이터만 활용"),
    min_peer_points: int = Query(2, ge=1, description="성장률 계산에 필요한 최소 일수(최소 2개 점)"),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user),
):
    """
    동일 직무(desired_job) 집단 기준:
      1) 지원자 수 (applicants)
      2) 1일 평균 성장점수 (내/동료 평균, 성장 배수 multiplier)
      3) 동일 직무 대비 상위 % (내 마지막 점수 기준, 동점 포함)

    공통 규칙
    - 하루 여러 세션이면 '그 날 마지막 세션'만 반영
    - 사용자별 날짜 순서를 dense_rank 로 1일차, 2일차...
    - peers 는 '동일 직무 + 나 제외'
    - cap_to_my_days=True 면 동료의 데이터도 내 학습 일수 이내까지만 사용(타임라인 정렬)
    """

    # 0) 내 직무
    my_job = db.execute(
        select(User.desired_job).where(User.id == user_id)
    ).scalar()
    if not my_job:
        raise HTTPException(status_code=400, detail="사용자의 desired_job 정보를 찾을 수 없습니다.")

    # 1) 윈도 컬럼: 캘린더 날짜 + (유저, 날짜)별 마지막 세션 + 유저별 일차
    session_date = func.date(InterviewSession.started_at).label("session_date")
    rn_in_day = func.row_number().over(
        partition_by=(InterviewSession.user_id, session_date),
        order_by=InterviewSession.started_at.desc()
    ).label("rn_in_day")
    day_index = func.dense_rank().over(
        partition_by=InterviewSession.user_id,
        order_by=session_date.asc()
    ).label("day_index")

    sess_subq = (
        select(
            InterviewSession.id.label("session_id"),
            InterviewSession.user_id.label("user_id"),
            session_date,
            rn_in_day,
            day_index
        ).subquery()
    )

    # 하루의 마지막 세션만
    last_of_day_subq = (
        select(
            sess_subq.c.session_id,
            sess_subq.c.user_id,
            sess_subq.c.session_date,
            sess_subq.c.day_index
        )
        .where(sess_subq.c.rn_in_day == 1)
        .subquery()
    )

    # 마지막 세션 + 점수 + 직무
    joined = (
        select(
            last_of_day_subq.c.user_id,
            last_of_day_subq.c.day_index,
            FinalReportSummary.total_score,
            User.desired_job
        )
        .join(FinalReportSummary, FinalReportSummary.session_id == last_of_day_subq.c.session_id)
        .join(User, User.id == last_of_day_subq.c.user_id)
        .where(FinalReportSummary.total_score.isnot(None))
        .subquery()
    )

    # 2) 내 학습 일수(최대 day_index)
    my_days = db.execute(
        select(func.max(joined.c.day_index)).where(joined.c.user_id == user_id)
    ).scalar()
    if not my_days or my_days < 1:
        raise HTTPException(status_code=404, detail="사용자의 학습 일수가 없습니다.")

    # 3) 동일 직무 데이터 로드 (cap 옵션 적용)
    filters = [joined.c.desired_job == my_job]
    if cap_to_my_days:
        filters.append(joined.c.day_index <= my_days)

    rows: List[Tuple[int, int, int]] = db.execute(
        select(joined.c.user_id, joined.c.day_index, joined.c.total_score).where(and_(*filters))
    ).all()

    if not rows:
        raise HTTPException(status_code=404, detail="동일 직무 데이터가 없습니다.")

    # 4) 사용자별 시계열 구성
    from collections import defaultdict
    by_user: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    for uid, d, s in rows:
        by_user[int(uid)].append((int(d), int(s)))

    def growth_per_day(points: List[Tuple[int,int]]) -> Optional[float]:
        # points: [(day_index, score), ...]
        if not points:
            return None
        pts = sorted(points, key=lambda x: x[0])
        first_day, first_score = pts[0]
        last_day, last_score = pts[-1]
        span = last_day - first_day
        if span < 1:
            return None  # 일차가 1개면 성장률 계산 불가
        return (last_score - first_score) / float(span)

    # 5) 내 성장률 & 동료 성장률 평균
    my_points = by_user.get(int(user_id), [])
    if not my_points:
        raise HTTPException(status_code=404, detail="사용자의 점수 시계열이 없습니다.")

    my_growth = growth_per_day(my_points)

    # 지원자 수(동일 직무 사용자 수, 적어도 1개 점수 있는 사람 기준)
    applicants = len({uid for uid, pts in by_user.items() if len(pts) > 0})

    # peers = 동일 직무 + 나 제외 + (최소 min_peer_points 데이터)
    peer_rates: List[float] = []
    for uid, pts in by_user.items():
        if uid == int(user_id):
            continue
        if len(pts) < min_peer_points:
            continue
        gp = growth_per_day(pts)
        if gp is not None:
            peer_rates.append(gp)

    peer_avg_growth = float(sum(peer_rates) / len(peer_rates)) if peer_rates else None
    multiplier = None
    if my_growth is not None and peer_avg_growth not in (None, 0.0):
        multiplier = my_growth / peer_avg_growth

    # 6) 동일 직무 대비 상위 % (마지막 점수 기준, 동점 포함)
    #    cap_to_my_days=True 이면 모든 사용자도 내 일수 이내에서의 '마지막 점수'를 사용
    latest_score_by_user: Dict[int, int] = {}
    for uid, pts in by_user.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        latest_score_by_user[uid] = pts_sorted[-1][1]  # 마지막 점수

    my_latest = latest_score_by_user.get(int(user_id))
    if my_latest is None:
        raise HTTPException(status_code=404, detail="사용자의 마지막 점수가 없습니다.")

    latest_scores = list(latest_score_by_user.values())
    population = len(latest_scores)  # 동일 직무 인원 수(최대 내 일수 고려 여부에 따라 위에서 이미 cap 적용됨)
    if population < 1:
        raise HTTPException(status_code=404, detail="동일 직무의 최신 점수 표본이 없습니다.")

    higher_or_equal = sum(1 for s in latest_scores if s >= my_latest)
    rank_percent = round((higher_or_equal / population) * 100.0, 2)

    return {
        "success": True,
        "job": my_job,
        "applicants": applicants,  # 동일 직무 지원자 수
        "cap_to_my_days": bool(cap_to_my_days),
        "min_peer_points": int(min_peer_points),
        "my_days": int(my_days),

        # 성장 관련
        "my_growth_per_day": round(my_growth, 2) if my_growth is not None else None,
        "peer_avg_growth_per_day": round(peer_avg_growth, 2) if peer_avg_growth is not None else None,
        "multiplier": round(multiplier, 2) if multiplier is not None else None,
        "note_growth": "growth_per_day = (마지막점수-처음점수)/(마지막일차-처음일차); 하루 여러세션 중 마지막 세션 점수 사용.",

        # 동일 직무 대비 상위 % (마지막 점수 기준)
        "job_rank": {
            "basis": "latest_score_within_my_days" if cap_to_my_days else "latest_score_all_days",
            "my_score": int(my_latest),
            "rank_percent": rank_percent,        # 상위 X%
            "population": int(population),
            "higher_or_equal": int(higher_or_equal)  # 내 점수 이상 인원(동점 포함)
        }
    }