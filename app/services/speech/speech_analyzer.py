from typing import Dict, List, Tuple
import librosa
import numpy as np
import re

class SpeechAnalyzer:
    def __init__(self, result: Dict):
        """
        :param result: ClovaSpeechClient의 response.json() 결과
        :param filler_words: 간투어 리스트 (예: ['어', '음', '아', '그', ...])
        """
        self.result = result
        self.segments = result.get("segments", [])
        self.total_text = ""
        self.total_duration = 0.0
        self.filler_words = [
            "음", "어", "아", "그", "저", "뭐", "이제", "그러니까", "있잖아요", "뭐랄까",
            "뭔가", "약간", "그니까", "뭐지", "어떻게", "뭐라고 해야 하지", "어떻게 보면",
            "사실", "약간은", "그런데", "근데", "그러면", "그런가", "뭐랄까요", "아마도",
            "혹시", "일단", "다만", "결국", "음...", "그…", "어…", "음… 그니까"
        ]

    def find_filler_words(self) -> List[Tuple[str, int]]:
        """
        전체 텍스트에서 간투어를 (간투어, 위치)로 반환
        """
        # 1. 구두점 제거
        cleaned_text = re.sub(r'[^\w\s]', '', self.total_text)

        # 2. 토큰화
        tokens = cleaned_text.strip().split()

        # 3. 간투어 탐지
        return [(word, idx) for idx, word in enumerate(tokens) if word in self.filler_words]

    def speech_speed_calculate(self) -> Dict:
        """
        내부 상태(self.segments)를 기반으로 말속도 분석 수행
        """
        for segment in self.segments:
            text = segment.get("text", "")
            start = float(segment.get("start", 0)) / 1000
            end = float(segment.get("end", 0)) / 1000
            duration = end - start

            self.total_text += text
            self.total_duration += duration

        if self.total_duration == 0:
            return {
                "syllables_per_min": 0.0,
                "words_per_min": 0.0,
                "total_duration_sec": 0.0,
                "total_text": ""
            }

        syllable_count = len(self.total_text.replace(" ", ""))
        word_count = len(self.total_text.strip().split())

        syllables_per_min = (syllable_count / self.total_duration) * 60
        words_per_min = (word_count / self.total_duration) * 60

        return {
            "syllables_per_min": round(syllables_per_min, 2),
            "words_per_min": round(words_per_min, 2),
            "total_duration_sec": round(self.total_duration, 2),
            "total_text": self.total_text.strip()
        }
    
    def calculate_pitch_variation(self, wav_path: str) -> Dict:
        """
        librosa로 음성의 pitch 변화 폭 분석 (단조로움 여부 판단)
        """
        try:
            # 1. librosa로 오디오 로드 (샘플링 주파수 16kHz로 맞춤)
            y, sr = librosa.load(wav_path, sr=16000)

            # 2. pyin으로 주파수(Hz) 추출 (fmin, fmax 설정은 사람이 낼 수 있는 범위 기준)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

            # 3. NaN 제거 (비발화 구간 제외)
            f0_filtered = f0[~np.isnan(f0)]

            if len(f0_filtered) < 30:
                return {"pitch_feedback": "음성이 짧아 분석 불가", "pitch_std": 0.0}

            # 4. 30프레임 단위로 표준편차 분석
            for i in range(0, len(f0_filtered) - 30 + 1, 30):
                window = f0_filtered[i:i + 30]
                std = np.std(window)
                if std < 10:
                    return {"pitch_feedback": "단조로운 음조로 들릴 수 있어요", "pitch_std": round(std, 2)}

            return {"pitch_feedback": "음조 변화가 적절합니다", "pitch_std": round(np.std(f0_filtered), 2)}

        except Exception as e:
            return {"pitch_feedback": f"pitch 분석 실패: {str(e)}", "pitch_std": 0.0}
    
    def find_filler_words(self, text) -> List[Tuple[str, int]]:
        """
        전체 텍스트에서 간투어를 (간투어, 위치)로 반환
        """
        # 1. 구두점 제거
        cleaned_text = re.sub(r'[^\w\s]', '', text)

        # 2. 토큰화
        tokens = cleaned_text.strip().split()

        # 3. 간투어 탐지
        return [(word, idx) for idx, word in enumerate(tokens) if word in self.filler_words]


if __name__ == "__main__":
    from app.services.stt.clovaspeech import ClovaSpeechClient
    from app.services.stt.vitospeech import VitoSpeechClient

    filler_words = [
    "음", "어", "아", "그", "저", "뭐", "이제", "그러니까", "있잖아요", "뭐랄까",
    "뭔가", "약간", "그니까", "뭐지", "어떻게", "뭐라고 해야 하지", "어떻게 보면",
    "사실", "약간은", "그런데", "근데", "그러면", "그런가", "뭐랄까요", "아마도",
    "혹시", "일단", "다만", "결국", "음...", "그…", "어…", "음… 그니까"
    ]

    clova = ClovaSpeechClient()
    vito = VitoSpeechClient()
    wav_file = r'C:\Users\UserK\AppData\Local\Temp\tmp63ci_9kv.wav'
    response = clova.req_upload(file=wav_file, completion="sync")
    text = vito.get_full_text_from_file(wav_file)
    result = response.json()

    analyzer = SpeechAnalyzer(result)
    speed_result = analyzer.speech_speed_calculate()
    pitch_result = analyzer.calculate_pitch_variation(wav_file)
    filler_result = analyzer.find_filler_words(text)

    print("📝 전체 텍스트:", speed_result["total_text"])
    print("🕒 발화 시간:", speed_result["total_duration_sec"], "초")
    print("💬 음절 기준 속도:", speed_result["syllables_per_min"], "음절/분")
    print("📖 단어 기준 속도:", speed_result["words_per_min"], "단어/분")
    print("🎼 음조 피드백:", pitch_result["pitch_feedback"])
    print("📊 표준편차:", pitch_result["pitch_std"], "Hz")
    print("🔍 간투어 탐지:", filler_result)