import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import json
import requests
import time
from app.core.jwt_token_updater import JwtTokenManager
from pathlib import Path

class VitoSpeechClient:
    def __init__(self, token_path: str = None):
        # 현재 파일 기준으로 루트 경로를 계산
        if token_path is None:
            BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 예: app/services/stt/ → app
            token_path = BASE_DIR / "vito_jwt_token.json"
        else:
            token_path = Path(token_path)

        JwtTokenManager().update_token_if_needed()

        if not token_path.exists():
            raise FileNotFoundError(f"{token_path} 파일이 존재하지 않습니다.")

        with open(token_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        self.jwt_token = token_data.get("access_token")
        if not self.jwt_token:
            raise ValueError("JWT 토큰을 JSON에서 읽을 수 없습니다.")

        self.base_url = "https://openapi.vito.ai/v1"

    def transcribe_file(self, file_path: str) -> str:
        """
        파일을 업로드하고 전사 요청을 보낸 후, transcribe_id를 반환
        """
        config = {
            "use_diarization": True,
            "diarization": {
                "spk_count": 1
            },
            "use_itn": False,
            "use_disfluency_filter": False,
            "use_profanity_filter": False,
            "use_paragraph_splitter": True,
            "paragraph_splitter": {
                "max": 50
            }
        }

        headers = {
            'Authorization': f'Bearer {self.jwt_token}'
        }

        files = {
            'file': open(file_path, 'rb')
        }

        data = {
            'config': json.dumps(config)
        }

        response = requests.post(
            f"{self.base_url}/transcribe",
            headers=headers,
            data=data,
            files=files
        )
        response.raise_for_status()

        result = response.json()
        print("📤 전송 결과:", result)
        return result["id"]

    def get_transcription_result(self, transcribe_id: str) -> dict:
        """
        전사 결과를 ID로 조회
        """
        headers = {
            'Authorization': f'bearer {self.jwt_token}'
        }

        response = requests.get(
            f"{self.base_url}/transcribe/{transcribe_id}",
            headers=headers
        )
        response.raise_for_status()

        result = response.json()
        print("📥 전사 결과:", result)
        return result

    def get_full_text_from_file(self, file_path: str, retry: int = 10, delay: int = 3) -> str:
        """
        파일을 업로드하고 전사 완료까지 대기한 후 전체 텍스트를 반환
        :param file_path: 음성 파일 경로
        :param retry: 최대 시도 횟수
        :param delay: 각 시도 사이 대기 시간 (초)
        :return: 전사된 전체 텍스트 문자열
        """
        # 1. 파일 업로드 및 transcribe_id 생성
        transcribe_id = self.transcribe_file(file_path)

        # 2. 전사 상태가 완료될 때까지 반복 확인
        for i in range(retry):
            result = self.get_transcription_result(transcribe_id)
            status = result.get("status")
            print(f"⌛️ 현재 상태: {status}")

            if status == "completed":
                # 3. 결과가 있으면 utterances 리스트에서 msg만 추출
                utterances = result.get('results', {}).get('utterances', [])
                messages = [utt.get('msg', '') for utt in utterances]
                full_text = " ".join(messages)
                return full_text

            time.sleep(delay)

        print("❌ 전사 완료되지 않음. 나중에 다시 시도하세요.")
        return ""

if __name__ == "__main__":
    client = VitoSpeechClient()
    transcribe_id = client.transcribe_file(r'C:\Users\ankh1\AppData\Local\Temp\tmp56u2svr7.wav')
    # 상태가 완료될 때까지 반복해서 확인
    for i in range(10):  # 최대 10번 (약 30초)
        result = client.get_transcription_result(transcribe_id)
        status = result.get("status")

        print(f"⌛️ 현재 상태: {status}")
        if status == "completed":
            break

        time.sleep(3)
    # msg만 추출
    utterances = result['results']['utterances']
    messages = [utterance['msg'] for utterance in utterances]

    # 하나의 자연스러운 텍스트로 이어붙이기
    full_text = " ".join(messages)
    if result.get("status") == "completed":
        print("📝 전사 결과 텍스트:", full_text)
    else:
        print("❌ 전사 완료되지 않음. 나중에 다시 시도하세요.")