# app/services/stt/stt_service.py

from app.services.stt.clovaspeech import ClovaSpeechClient
from app.services.stt.vitospeech import VitoSpeechClient
from typing import Tuple

class STTService:
    def __init__(self, stt_type: str = "clova"):
        """
        stt_type: "clova" 또는 "vito"
        """
        self.stt_type = stt_type.lower()

        if self.stt_type == "clova":
            self.client = ClovaSpeechClient()
        elif self.stt_type == "vito":
            self.client = VitoSpeechClient()
        else:
            raise ValueError(f"지원하지 않는 STT 타입: {stt_type}")

    def transcribe(self, wav_path: str) -> Tuple[str, dict]:
        """
        wav 파일 경로를 입력받아 텍스트와 분석용 결과 JSON 반환
        """
        if self.stt_type == "clova":
            response = self.client.req_upload(file=wav_path, completion="sync")
            result = response.json()
            segments = result.get("segments", [])
            text = " ".join(seg.get("text", "") for seg in segments)
            return text, result

        elif self.stt_type == "vito":
            text = self.client.get_full_text_from_file(wav_path)
            result = {"segments": [{"text": text, "start": 0, "end": 0}]}  # 형식 통일
            return text, result
