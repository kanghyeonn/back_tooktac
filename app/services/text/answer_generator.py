from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class ModelAnswerGenerator:

    def __init__(self):
        self.client = OpenAI()

    def generate_technical_answer(self, question: str) -> str:
        system_msg = {
            "role": "system",
            "content": """당신은 10년 경력의 시니어 개발자이자 기술 면접관입니다. 
정확하고 실무적인 관점에서 간결하게 답변하세요."""
        }

        prompt = f"""
기술 면접 질문: {question}

정확히 3문장으로 답변하세요.

구조: 1.정의 2.특징/차이점 3.실무관점/사용예시

Few-shot Examples:

Q: "프로세스와 스레드의 차이를 설명해주세요"
A: "프로세스는 독립 메모리를 가진 실행 단위입니다. 스레드는 프로세스 내 자원을 공유하는 경량 실행 단위입니다. 멀티스레딩은 성능상 유리하나 동기화 이슈에 주의해야 합니다."

Q: "REST API와 GraphQL의 차이점을 설명해주세요"
A: "REST는 HTTP 메서드 기반의 리소스 중심 API 설계 방식입니다. GraphQL은 클라이언트가 필요한 데이터만 요청할 수 있는 쿼리 언어입니다. GraphQL은 유연하지만 캐싱과 보안 설정이 더 복잡합니다."

Q: "HTTP와 HTTPS의 차이를 설명해주세요"
A: "HTTP는 웹에서 데이터를 주고받는 기본 통신 프로토콜입니다. HTTPS는 HTTP에 SSL/TLS 암호화를 추가한 보안 프로토콜입니다. HTTPS는 데이터 보안이 중요한 서비스에서 필수적으로 사용됩니다."

3문장만 출력:
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                system_msg,
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 문장 수 제한 준수를 위해 낮춤
            max_tokens=300,  # 3문장 제한을 위해 토큰 수 감소
            top_p=0.9  # 품질 높은 답변 선택을 위한 추가
        )

        output = response.choices[0].message.content.strip()
        return output

    def generate_situational_answer(self, question: str, user_answer: str) -> str:
        system_msg = {
            "role": "system",
            "content": """당신은 HR 전문가이자 면접 코칭 전문가입니다. 
지원자 답변의 핵심 키워드와 경험 내용은 절대 변경하지 말고, 표현 방식만 개선하여 더 구체적이고 설득력 있게 만드세요."""
        }

        prompt = f"""
상황형 면접 질문: {question}
지원자 답변: {user_answer}

지원자의 핵심 키워드와 경험을 유지하며 정확히 3문장으로 개선하세요.

구조: 1.상황 2.행동 3.결과+배운점

Few-shot Examples:

원본: "팀원과 갈등이 있었는데 대화로 해결했습니다."
개선: "팀원과의 의견 차이로 갈등 상황이 발생했습니다. 1:1 대화를 통해 서로의 입장을 이해하고 합의점을 찾았습니다. 갈등이 해결되어 팀워크가 개선되었고, 소통의 중요성을 깨달았습니다."

원본: "일정이 부족해서 야근을 많이 했어요."
개선: "프로젝트 마감일이 촉박한 상황에 직면했습니다. 업무 우선순위를 재조정하고 팀원들과 역할을 분담해 집중적으로 작업했습니다. 마감일을 준수하여 완료했고, 사전 계획과 팀워크의 중요성을 배웠습니다."

원본: "고객 불만이 있어서 빨리 처리했습니다."
개선: "고객으로부터 서비스 품질에 대한 불만 사항이 접수되었습니다. 즉시 고객과 통화하여 문제를 파악하고 24시간 내 해결책을 제시했습니다. 고객 만족도가 향상되었고, 선제적 대응의 가치를 체감했습니다."

3문장만 출력:
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                system_msg,
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 문장 수 제한 준수를 위해 낮춤
            max_tokens=300,  # 3문장을 위해 토큰 수 증가
            top_p=0.9
        )

        output = response.choices[0].message.content.strip()
        return output

    def generate(self, question: str, user_answer: str, evaluation_type: str) -> str:
        if evaluation_type == "technical":
            return self.generate_technical_answer(question)
        else:
            return self.generate_situational_answer(question, user_answer)