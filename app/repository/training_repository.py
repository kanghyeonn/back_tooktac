from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.models.interview import InterviewSession
from app.models.report import FinalReportSummary
from app.models.user import User
from typing import Optional


# 1. 일차별 평균 점수 (모든 사용자 대상)
def get_day_index_avg_scores(db: Session, days: int):
    session_date = func.date(InterviewSession.started_at).label("session_date")

    rn_in_day = func.row_number().over(
        partition_by=(InterviewSession.user_id, session_date),
        order_by=InterviewSession.started_at.desc()
    ).label("rn_in_day")

    day_index = func.dense_rank().over(
        partition_by=InterviewSession.user_id,
        order_by=session_date.asc()
    ).label("day_index")

    sess_subq = select(
        InterviewSession.id.label("session_id"),
        InterviewSession.user_id.label("user_id"),
        session_date,
        rn_in_day,
        day_index
    ).subquery()

    last_of_day_subq = select(
        sess_subq.c.session_id,
        sess_subq.c.user_id,
        sess_subq.c.session_date,
        sess_subq.c.day_index
    ).where(sess_subq.c.rn_in_day == 1).subquery()

    joined = select(
        last_of_day_subq.c.day_index,
        FinalReportSummary.total_score
    ).join(
        FinalReportSummary,
        FinalReportSummary.session_id == last_of_day_subq.c.session_id
    ).where(FinalReportSummary.total_score.isnot(None)).subquery()

    avg_query = select(
        joined.c.day_index,
        func.avg(joined.c.total_score).label("avg_score")
    ).where(joined.c.day_index <= days
    ).group_by(joined.c.day_index).order_by(joined.c.day_index.asc())

    return db.execute(avg_query).all()


# 2. 로그인 사용자의 일차까지 다른 사용자들의 평균 점수
def get_peer_avg_scores_up_to_my_days(db: Session, user_id: int):
    session_date = func.date(InterviewSession.started_at).label("session_date")

    rn_in_day = func.row_number().over(
        partition_by=(InterviewSession.user_id, session_date),
        order_by=InterviewSession.started_at.desc()
    ).label("rn_in_day")

    day_index = func.dense_rank().over(
        partition_by=InterviewSession.user_id,
        order_by=session_date.asc()
    ).label("day_index")

    sess_subq = select(
        InterviewSession.id.label("session_id"),
        InterviewSession.user_id.label("user_id"),
        session_date,
        rn_in_day,
        day_index
    ).subquery()

    last_of_day_subq = select(
        sess_subq.c.session_id,
        sess_subq.c.user_id,
        sess_subq.c.session_date,
        sess_subq.c.day_index
    ).where(sess_subq.c.rn_in_day == 1).subquery()

    my_day_query = select(func.max(last_of_day_subq.c.day_index)).where(
        last_of_day_subq.c.user_id == user_id
    )
    my_days = db.execute(my_day_query).scalar()

    joined = select(
        last_of_day_subq.c.day_index,
        FinalReportSummary.total_score
    ).join(
        FinalReportSummary,
        FinalReportSummary.session_id == last_of_day_subq.c.session_id
    ).where(
        FinalReportSummary.total_score.isnot(None),
        last_of_day_subq.c.user_id != user_id,
        last_of_day_subq.c.day_index <= my_days
    ).subquery()

    agg = select(
        joined.c.day_index,
        func.count(joined.c.total_score).label("n"),
        func.avg(joined.c.total_score).label("avg_score")
    ).group_by(joined.c.day_index).order_by(joined.c.day_index.asc())

    rows = db.execute(agg).all()

    return {"my_days": my_days, "rows": rows}


# 3. 사용자별 최신 세션의 점수 (윈도우 함수 사용)
def get_latest_total_scores(db: Session):
    rn = func.row_number().over(
        partition_by=InterviewSession.user_id,
        order_by=InterviewSession.started_at.desc()
    ).label("rn")

    sess_subq = select(
        InterviewSession.id.label("session_id"),
        InterviewSession.user_id,
        InterviewSession.started_at,
        rn
    ).subquery()

    q = select(FinalReportSummary.total_score).join(
        sess_subq, sess_subq.c.session_id == FinalReportSummary.session_id
    ).where(
        sess_subq.c.rn == 1,
        FinalReportSummary.total_score.isnot(None)
    )

    return [r[0] for r in db.execute(q).all()]


# 4. 사용자별 최신 세션 점수 (윈도우 미지원 대안)
def get_latest_total_scores_fallback(db: Session):
    latest_started_subq = select(
        InterviewSession.user_id,
        func.max(InterviewSession.started_at).label("max_started_at")
    ).group_by(InterviewSession.user_id).subquery()

    latest_session_subq = select(
        InterviewSession.id.label("session_id"),
        InterviewSession.user_id
    ).join(
        latest_started_subq,
        (InterviewSession.user_id == latest_started_subq.c.user_id) &
        (InterviewSession.started_at == latest_started_subq.c.max_started_at)
    ).subquery()

    q = select(FinalReportSummary.total_score).join(
        latest_session_subq,
        latest_session_subq.c.session_id == FinalReportSummary.session_id
    ).where(FinalReportSummary.total_score.isnot(None))

    return [r[0] for r in db.execute(q).all()]


# 5. 특정 직무 대상 사용자들의 점수 시계열
def get_job_scores_by_user(
    db: Session, desired_job: str, exclude_user_id: int, max_day: int, cap: bool = True
):
    session_date = func.date(InterviewSession.started_at).label("session_date")
    rn_in_day = func.row_number().over(
        partition_by=(InterviewSession.user_id, session_date),
        order_by=InterviewSession.started_at.desc()
    ).label("rn_in_day")
    day_index = func.dense_rank().over(
        partition_by=InterviewSession.user_id,
        order_by=session_date.asc()
    ).label("day_index")

    sess_subq = select(
        InterviewSession.id.label("session_id"),
        InterviewSession.user_id.label("user_id"),
        session_date,
        rn_in_day,
        day_index
    ).subquery()

    last_of_day_subq = select(
        sess_subq.c.session_id,
        sess_subq.c.user_id,
        sess_subq.c.session_date,
        sess_subq.c.day_index
    ).where(sess_subq.c.rn_in_day == 1).subquery()

    joined = select(
        last_of_day_subq.c.user_id,
        last_of_day_subq.c.day_index,
        FinalReportSummary.total_score,
        User.desired_job
    ).join(
        FinalReportSummary,
        FinalReportSummary.session_id == last_of_day_subq.c.session_id
    ).join(
        User,
        User.id == last_of_day_subq.c.user_id
    ).where(
        FinalReportSummary.total_score.isnot(None),
        User.desired_job == desired_job
    )

    if cap:
        joined = joined.where(last_of_day_subq.c.day_index <= max_day)

    return db.execute(joined).all()


# 6. 사용자 desired_job 조회
def get_user_desired_job(db: Session, user_id: int) -> Optional[str]:
    q = select(User.desired_job).where(User.id == user_id)
    return db.execute(q).scalar()


# 7. 사용자 최대 day_index 조회
def get_user_max_day_index(db: Session, user_id: int) -> Optional[int]:
    session_date = func.date(InterviewSession.started_at).label("session_date")
    rn_in_day = func.row_number().over(
        partition_by=(InterviewSession.user_id, session_date),
        order_by=InterviewSession.started_at.desc()
    ).label("rn_in_day")
    day_index = func.dense_rank().over(
        partition_by=InterviewSession.user_id,
        order_by=session_date.asc()
    ).label("day_index")

    sess_subq = select(
        InterviewSession.id.label("session_id"),
        InterviewSession.user_id.label("user_id"),
        session_date,
        rn_in_day,
        day_index
    ).subquery()

    last_of_day_subq = select(
        sess_subq.c.session_id,
        sess_subq.c.user_id,
        sess_subq.c.session_date,
        sess_subq.c.day_index
    ).where(sess_subq.c.rn_in_day == 1).subquery()

    q = select(func.max(last_of_day_subq.c.day_index)).where(
        last_of_day_subq.c.user_id == user_id
    )

    return db.execute(q).scalar()
