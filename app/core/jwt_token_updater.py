import requests
import json
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

class JwtTokenManager:
    def __init__(self, output_path=None):
        load_dotenv()
        self.client_id = os.getenv("VITO_CLIENT_ID")
        self.client_secret = os.getenv("VITO_CLIENT_SECRET")
        self.client_id = os.getenv("VITO_CLIENT_ID")
        self.client_secret = os.getenv("VITO_CLIENT_SECRET")
        self.auth_url = "https://openapi.vito.ai/v1/authenticate"

        if output_path is None:
            BASE_DIR = Path(__file__).resolve().parent.parent  # 예: app/core
            self.output_path = BASE_DIR / "vito_jwt_token.json"
        else:
            self.output_path = Path(output_path)

    def fetch_token(self) -> dict:
        """API를 통해 새 JWT 토큰을 요청하고 반환"""
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        headers={"Accept": "application/json"}
        print("🔐 JWT 토큰 요청 중...")
        response = requests.post(self.auth_url, data=payload, headers=headers)
        response.raise_for_status()

        token_data = response.json()
        token_data["fetched_at"] = datetime.now().isoformat()
        token_data["expires_at"] = (datetime.now() + timedelta(hours=6)).isoformat()

        return token_data

    def save_token(self, token_data: dict):
        """토큰 정보를 JSON 파일로 저장"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(token_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 토큰 저장 완료: {self.output_path}")

    def load_token(self) -> dict:
        """저장된 토큰 파일 로드 (없으면 None)"""
        if not os.path.exists(self.output_path):
            return None

        with open(self.output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def is_token_expired(self, token_data: dict) -> bool:
        """토큰 만료 여부 판단"""
        expires_at = datetime.fromisoformat(token_data.get("expires_at"))
        return datetime.now() >= expires_at

    def update_token_if_needed(self):
        """토큰 만료 시 재발급"""
        current_token = self.load_token()

        if current_token is None or self.is_token_expired(current_token):
            print("🔄 토큰이 만료되었거나 존재하지 않음. 새로 발급합니다.")
            new_token = self.fetch_token()
            self.save_token(new_token)
        else:
            print("🟢 현재 토큰은 유효합니다.")


if __name__ == "__main__":
    manager = JwtTokenManager()
    manager.update_token_if_needed()

