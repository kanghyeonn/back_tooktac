# app/utils/scoring.py

class QuestionTypeWeights:
    """질문 유형별 가중치 시스템"""

    WEIGHTS = {
        "개념설명형": {"text": 0.5, "voice": 0.2, "emotion": 0.1, "video": 0.2},
        "기술형": {"text": 0.5, "voice": 0.3, "emotion": 0.1, "video": 0.1},
        "상황형": {"text": 0.4, "voice": 0.15, "emotion": 0.25, "video": 0.2},
        "행동형": {"text": 0.4, "voice": 0.2, "emotion": 0.2, "video": 0.2},
    }

    @classmethod
    def calculate_weighted_score(cls, question_analysis: dict) -> float:
        question_type = question_analysis.get('type', '개념설명형')
        detail_analysis = question_analysis.get('detailAnalysis', {})

        weights = cls.WEIGHTS.get(question_type, cls.WEIGHTS['개념설명형'])

        text_score = detail_analysis.get('text', {}).get('score', 0)
        voice_score = detail_analysis.get('voice', {}).get('score', 0)
        emotion_score = detail_analysis.get('emotion', {}).get('score', 0)
        video_score = detail_analysis.get('video', {}).get('score', 0)

        weighted_score = (
            text_score * weights['text'] +
            voice_score * weights['voice'] +
            emotion_score * weights['emotion'] +
            video_score * weights['video']
        )

        return round(weighted_score, 1)
