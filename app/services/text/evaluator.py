from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()

class AnswerEvaluator:
    INTERVIEWER_PERSONA = """
당신은 다음과 같은 배경을 가진 전문가입니다:
- 20년 경력의 시니어 면접관 및 인사담당자
- 다양한 직무 면접 경험 보유
- 수천 명의 후보자 평가 및 채용 결정 경험
- 우수 인재와 부적합 인재를 구분하는 직관력 보유
- 질문과 답변의 숨겨진 의도를 정확히 파악하는 능력
"""

    def __init__(self):
        self.client = OpenAI()

    def evaluate(self, question: str, user_answer: str, evaluation_type: str) -> dict:
        if evaluation_type == "technical":
            return self._evaluate_technical(question, user_answer)
        else:
            return self._evaluate_situational(question, user_answer)

    def _evaluate_technical(self, question: str, user_answer: str) -> dict:
        prompt = """
    """ + self.INTERVIEWER_PERSONA + """ 

    Return the result in the following JSON format (all content must be in Korean):

    {
    "intent_score": float,           // 질문 의도 파악 점수 (1점부터 10점까지)
    "knowledge_score": float,        // 지식 정확도 점수 (1점부터 10점까지)
    "strengths": ["강점1", "강점2"],
    "improvements": ["개선점1", "개선점2"],
    "final_feedback": "전체 총평 (in Korean)"
    }

    Scoring Criteria (1점부터 10점까지):

    1. **intent_score** (Understanding the Question's Intent)
    - 9-10점: 질문의 핵심 의도를 완벽히 파악하고 정확히 답변
    - 7-8점: 질문 의도를 잘 이해하고 적절히 답변
    - 5-6점: 질문 의도를 부분적으로 이해
    - 3-4점: 질문과 관련있지만 핵심을 놓침
    - 1-2점: 질문 의도와 거의 무관한 답변

    2. **knowledge_score** (Technical Accuracy and Depth)
    - 9-10점: 기술적으로 정확하고 상세하며 체계적인 지식 제시
    - 7-8점: 기본적인 이해를 보이지만 설명이 모호하거나 불완전
    - 5-6점: 기본 개념은 알고 있으나 깊이 부족
    - 3-4점: 부분적으로 올바르지만 오해나 부정확한 내용 포함
    - 1-2점: 잘못되거나 오해의 소지가 있는 기술 내용

    3. **strengths & improvements**
    - 각각 1-2개의 구체적인 의견을 한국어로 제시
    - 일반적인 칭찬이나 비판 지양, 구체적이고 도움되는 내용

    4. **final_feedback**
    - 실제 한국어 기술 면접관처럼 전문적이고 건설적인 한 줄 총평

    -----

    질문 (in Korean):  
    """ + question + """

    사용자 답변 (in Korean):  
    """ + user_answer + """

    Important: Output must be written in Korean only.  
    Strictly follow the JSON format above. Do not add any explanation or commentary outside the JSON.
    """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )

        result = json.loads(response.choices[0].message.content.strip())
        result["intent_score"] = max(1.0, min(10.0, float(result.get("intent_score", 1.0))))
        result["knowledge_score"] = max(1.0, min(10.0, float(result.get("knowledge_score", 1.0))))
        return result

    def _evaluate_situational(self, question: str, user_answer: str) -> dict:
        prompt = """
    """ + self.INTERVIEWER_PERSONA + """  

    You will receive an interview question and a candidate's answer, both in Korean.  
    Your task is to evaluate the answer across two dimensions and return a structured evaluation in JSON format.  
    **All output (including feedback and explanation) must be written in Korean. Do not use English.**

    Please strictly follow this output format:

    {
    "intent_score": float,           // 질문 의도 파악 점수 (1점부터 10점까지)
    "knowledge_score": float,        // 지식 정확도 점수 (1점부터 10점까지)
    "strengths": ["강점1", "강점2"],
    "improvements": ["개선점1", "개선점2"],
    "final_feedback": "전체 총평 (in Korean)"
    }

    Scoring Criteria (1점부터 10점까지):

    1. **intent_score** (Understanding the Question's Intent)
    - 9-10점: 질문의 핵심 의도를 완벽히 파악하고 정확히 답변
    - 7-8점: 질문 의도를 잘 이해하고 적절히 답변
    - 5-6점: 질문 의도를 부분적으로 이해
    - 3-4점: 질문과 관련있지만 핵심을 놓침
    - 1-2점: 질문 의도와 거의 무관한 답변

    2. **knowledge_score** (Accuracy of Knowledge and Experience)
    - 9-10점: 정확하고 구체적인 경험과 지식을 제시
    - 7-8점: 적절한 경험을 제시하지만 일부 모호한 부분 존재
    - 5-6점: 기본적인 이해는 보이지만 구체성 부족
    - 3-4점: 경험이나 지식이 부분적으로만 적절
    - 1-2점: 부정확하거나 관련성이 낮은 내용

    3. **strengths & improvements**
    - 각각 1-2개의 짧고 구체적인 의견을 한국어로 제시
    - 모호한 칭찬이나 일반적인 비판 지양, 구체적이고 도움되는 내용

    4. **final_feedback**
    - 실제 한국어 면접관처럼 간결한 전체 의견
    - 한국어로 작성하고, 존중하며 건설적으로

    -----

    질문 (in Korean):  
    """ + question + """

    사용자 답변 (in Korean):  
    """ + user_answer + """

    Repeat: Output must be written in Korean only. Output must strictly follow the JSON format above. Do not add extra explanation.
    """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )

        result = json.loads(response.choices[0].message.content.strip())
        result["intent_score"] = max(1.0, min(10.0, float(result.get("intent_score", 1.0))))
        result["knowledge_score"] = max(1.0, min(10.0, float(result.get("knowledge_score", 1.0))))
        return result