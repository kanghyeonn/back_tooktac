from app.services.text.answer_generator import ModelAnswerGenerator
from app.services.text.calculator import FinalScoreCalculator
from app.services.text.evaluator import AnswerEvaluator
from app.services.text.scorer import SimilarityScorer

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

class EvaluationOrchestrator:
    def __init__(self):
        self.model_answer_generator = ModelAnswerGenerator()
        self.similarity_scorer = SimilarityScorer()
        self.answer_evaluator = AnswerEvaluator()
        self.score_calculator = FinalScoreCalculator()

    def evaluate_answer(self, question: str, user_answer: str, question_type: str) -> dict:
        # Step 1: 입력 전처리
        processed_input = preprocess_input(question, user_answer, question_type)

        # Step 2: 모범답변 생성
        model_answer = self.model_answer_generator.generate(
            processed_input["question"],
            processed_input["user_answer"],
            processed_input["evaluation_type"]
        )

        # Step 3: 유사도 계산
        similarity_score = self.similarity_scorer.calculate_similarity(
            processed_input["user_answer"],
            model_answer
        )

        # Step 4: 답변 평가
        evaluation_result = self.answer_evaluator.evaluate(
            processed_input["question"],
            processed_input["user_answer"],
            processed_input["evaluation_type"]
        )

        # Step 5: 최종 점수 계산
        scores = {
            "similarity_score": similarity_score,
            "intent_score": evaluation_result["intent_score"],
            "knowledge_score": evaluation_result["knowledge_score"]
        }

        final_score = self.score_calculator.calculate_final_score(
            scores, processed_input["evaluation_type"]
        )

        # Step 6: 결과 반환
        return {
            "question": processed_input["question"],
            "user_answer": processed_input["user_answer"],
            "question_type": processed_input["question_type"],
            "evaluation_type": processed_input["evaluation_type"],
            "model_answer": model_answer,
            "similarity": similarity_score,
            "intent_score": evaluation_result["intent_score"],
            "knowledge_score": evaluation_result["knowledge_score"],
            "final_score": final_score,
            "feedback": {
                "strengths": evaluation_result["strengths"],
                "improvements": evaluation_result["improvements"],
                "final_feedback": evaluation_result["final_feedback"]
            }
        }