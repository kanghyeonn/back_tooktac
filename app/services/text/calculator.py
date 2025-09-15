class FinalScoreCalculator:
    WEIGHTS = {
        "technical": {"similarity": 0.2, "intent": 0.3, "knowledge": 0.5},
        "situational": {"similarity": 0.3, "intent": 0.35, "knowledge": 0.35}
    }

    def calculate_final_score(self, scores: dict, evaluation_type: str) -> int:
        # 가중치 적용 합산
        weights = self.WEIGHTS[evaluation_type]

        # 모든 점수를 0-1로 정규화
        similarity_norm = scores["similarity_score"]  # 이미 0-1
        intent_norm = (scores["intent_score"] - 1) / 9  # 1-10 → 0-1
        knowledge_norm = (scores["knowledge_score"] - 1) / 9  # 1-10 → 0-1

        final_score = (
                              similarity_norm * weights["similarity"] +
                              intent_norm * weights["intent"] +
                              knowledge_norm * weights["knowledge"]
                      ) * 100  # 0-1 → 0-100

        return round(final_score)


def preprocess_input(question_text: str, user_answer_text: str, question_type: str) -> dict:
    TYPE_MAPPING = {
        "개념설명형": "technical",
        "기술형": "technical",
        "상황형": "situational",
        "행동형": "situational"
    }

    return {
        "question": question_text.strip(),
        "user_answer": user_answer_text.strip(),
        "question_type": question_type,
        "evaluation_type": TYPE_MAPPING.get(question_type, "technical")
    }