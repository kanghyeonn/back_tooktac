# app/utils/auth_ws.py
from fastapi import WebSocket, status, WebSocketException
from app.core.security import decode_access_token
from jose import ExpiredSignatureError, JWTError, jwt
import os
from dotenv import load_dotenv

load_dotenv()

# async def get_user_id_from_websocket(websocket: WebSocket) -> int:
#     # 1. 쿠키에서 access_token 추출
#     token = websocket.cookies.get("access_token")
#     if not token:
#         raise WebSocketException(
#             code=status.WS_1008_POLICY_VIOLATION,
#             reason="Access token missing"
#         )

#     # 2. 토큰 디코딩
#     try:
#         user_id = decode_access_token(token)  # 내부적으로 exp 확인
#         return user_id

#     # 3. 만료된 토큰 처리
#     except ExpiredSignatureError:
#         raise WebSocketException(
#             code=4401,  # 사용자 정의 코드 (4000~4999 범위 권장)
#             reason="Token expired"
#         )

#     # 4. 기타 잘못된 토큰
#     except JWTError:
#         raise WebSocketException(
#             code=4403,
#             reason="Invalid token"
#         )


# app/utils/auth_ws.py

SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGS = ["HS256"]

async def get_user_id_from_websocket(ws: WebSocket) -> int:
    # 0) 진단 로그
    print("[AUTH_WS] cookies =", dict(ws.cookies))
    print("[AUTH_WS] headers.authorization =", ws.headers.get("authorization"))
    print("[AUTH_WS] query_params =", dict(ws.query_params))

    # 1) 쿠키 우선
    token = ws.cookies.get("access_token")

    # 2) Authorization: Bearer xxx
    if not token:
        auth = ws.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()

    # 3) ?token=xxx (디버깅/임시 허용)
    if not token:
        token = ws.query_params.get("token")

    if not token:
        raise ValueError("Access token missing")

    try:
        payload = jwt.decode(token, SECRET, algorithms=ALGS)
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}")

    sub = payload.get("sub")
    if sub is None:
        raise ValueError("Invalid token payload: sub missing")

    return int(sub)
