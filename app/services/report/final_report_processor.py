# %%
from google.cloud import documentai
from google.cloud import storage
from prettytable import PrettyTable
import google.generativeai as genai
import re
import json
from typing import Dict, List, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_api_key)

project_id = os.getenv("PROJECT_ID")
processor_id = os.getenv("PROCESSOR_ID")
location = os.getenv("LOCATION", "us")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


# %%
# data_parser

@dataclass
class UserInfo:
    user_id: str
    user_nickname: str
    interview_id: str
    interview_date: str
    interview_duration: int


@dataclass
class QuestionAnalysis:
    question_id: str
    name: str
    type: str
    final_score: int
    question: str
    my_answer: str
    model_answer: str
    detail_analysis: Dict[str, Dict[str, float]]  # text, voice, video, emotion
    feedback: str
    strengths: List[str]
    improvements: List[str]


class DataParser:
    """JSON 데이터를 파싱하고 검증하는 클래스"""

    def parse_interview_data(self, interview_data: Union[str, Dict, List]) -> Dict:
        """
        다양한 형태의 면접 데이터를 파싱하여 표준 형식으로 변환

        Args:
            interview_data: 면접 데이터 (다양한 형태)

        Returns:
            파싱된 데이터 딕셔너리
        """

        # 1. JSON 문자열인 경우 딕셔너리로 변환
        if isinstance(interview_data, str):
            try:
                interview_data = json.loads(interview_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"유효하지 않은 JSON 형식입니다: {str(e)}")

        # 2. 데이터 형태에 따라 처리 분기
        if isinstance(interview_data, list):
            # 리스트 형태: [{"question_id": "q1", ...}, {"question_id": "q2", ...}]
            return self._parse_list_format(interview_data)

        elif isinstance(interview_data, dict):
            # 딕셔너리 형태: {"user_info": {...}, "questions": [...]}
            return self._parse_dict_format(interview_data)

        else:
            raise ValueError(f"지원하지 않는 데이터 형태입니다: {type(interview_data)}")

    def _parse_list_format(self, questions_list: List[Dict]) -> Dict:
        """
        리스트 형태의 데이터 파싱

        형태: [
            {
                "question_id": "q1",
                "user_info": {"user_nickname": "김면접", ...},  # 첫 번째 질문에만 있을 수 있음
                "question_number": 1,
                "question_type": "개념설명형",
                "final_score": 88,




                "question_text": "질문 내용",
                "user_answer": "사용자 답변",
                "model_answer": "모범답안",
                "detail_analysis": {...},
                "feedback": "피드백",
                "strengths": ["강점1", "강점2"],
                "improvements": ["개선점1", "개선점2"]
            },
            {...}, # 질문 2~6
        ]
        """

        if not questions_list or len(questions_list) != 6:
            raise ValueError(f"질문 데이터는 정확히 6개여야 합니다. 현재: {len(questions_list)}개")

        # 사용자 정보 추출 (첫 번째 질문에서 또는 별도로 제공)
        user_info = self._extract_user_info(questions_list)

        # 질문별 분석 데이터 변환
        question_analyses = []
        for i, question_data in enumerate(questions_list):
            try:
                qa = self._convert_to_question_analysis(question_data, i + 1)
                question_analyses.append(qa)
            except Exception as e:
                raise ValueError(f"질문 {i + 1} 데이터 변환 실패: {str(e)}")

        return {
            'user_info': user_info,
            'question_analyses': question_analyses
        }

    def _parse_dict_format(self, data_dict: Dict) -> Dict:
        """
        딕셔너리 형태의 데이터 파싱

        형태: {
            "user_info": {
                "user_id": "user123",
                "user_nickname": "김면접",
                "interview_id": "interview456",
                "interview_date": "2024-08-04",
                "interview_duration": 25
            },
            "questions": [
                {"question_id": "q1", ...},
                {"question_id": "q2", ...},
                ...
            ]
        }
        """

        # 필수 키 검증
        if 'user_info' not in data_dict:
            raise ValueError("user_info 필드가 없습니다")

        questions_key = None
        for key in ['questions', 'question_analyses', 'question_scores']:
            if key in data_dict:
                questions_key = key
                break

        if not questions_key:
            raise ValueError("질문 데이터 필드를 찾을 수 없습니다 (questions, question_analyses, question_scores 중 하나 필요)")

        # 사용자 정보 변환
        user_info = self._convert_to_user_info(data_dict['user_info'])

        # 질문 데이터 변환
        questions_list = data_dict[questions_key]
        # if len(questions_list) != 6:
        #     raise ValueError(f"질문 데이터는 정확히 6개여야 합니다. 현재: {len(questions_list)}개")

        question_analyses = []
        for i, question_data in enumerate(questions_list):
            qa = self._convert_to_question_analysis(question_data, i + 1)
            question_analyses.append(qa)

        return {
            'user_info': user_info,
            'question_analyses': question_analyses
        }

    def _extract_user_info(self, questions_list: List[Dict]) -> UserInfo:
        """리스트 형태 데이터에서 사용자 정보 추출"""

        # 첫 번째 질문에서 사용자 정보 찾기
        first_question = questions_list[0]

        if 'user_info' in first_question:
            return self._convert_to_user_info(first_question['user_info'])

        # user_info가 별도로 없으면 개별 필드에서 추출
        user_data = {}
        for field in ['user_id', 'user_nickname', 'interview_id', 'interview_date', 'interview_duration']:
            if field in first_question:
                user_data[field] = first_question[field]

        if not user_data:
            # 기본값 설정
            user_data = {
                'user_id': 'unknown',
                'user_nickname': '면접자',
                'interview_id': 'interview_unknown',
                'interview_date': '2024-08-04',
                'interview_duration': 30
            }

        return self._convert_to_user_info(user_data)

    def _convert_to_user_info(self, user_data: Dict) -> UserInfo:
        """딕셔너리를 UserInfo 객체로 변환"""
        return UserInfo(
            user_id=user_data.get('user_id', 'unknown'),
            user_nickname=user_data.get('user_nickname', '면접자'),
            interview_id=user_data.get('interview_id', 'interview_unknown'),
            interview_date=user_data.get('interview_date', '2024-08-04'),
            interview_duration=user_data.get('interview_duration', 30)
        )

    def _convert_to_question_analysis(self, question_data: Dict, question_number: int) -> QuestionAnalysis:
        """딕셔너리를 QuestionAnalysis 객체로 변환"""

        # 필수 필드 검증
        required_fields = ['final_score', 'question_text', 'user_answer', 'detail_analysis']
        for field in required_fields:
            if field not in question_data and field.replace('_', '') not in question_data:
                # 대체 필드명도 확인
                alt_names = {
                    'final_score': ['score', 'total_score'],
                    'question_text': ['question', 'question_content'],
                    'user_answer': ['my_answer', 'answer'],
                    'detail_analysis': ['analysis', 'detailed_analysis']
                }

                found = False
                if field in alt_names:
                    for alt_name in alt_names[field]:
                        if alt_name in question_data:
                            question_data[field] = question_data[alt_name]
                            found = True
                            break

                if not found:
                    raise ValueError(f"필수 필드가 없습니다: {field}")

        # QuestionAnalysis 객체 생성
        return QuestionAnalysis(
            question_id=question_data.get('question_id', f'q{question_number}'),
            name=question_data.get('name', f'질문 {question_number}'),
            type=question_data.get('question_type', question_data.get('type', '일반형')),
            final_score=int(question_data['final_score']),
            question=question_data['question_text'],
            my_answer=question_data['user_answer'],
            model_answer=question_data.get('model_answer', '모범답안이 없습니다.'),
            detail_analysis=question_data['detail_analysis'],
            feedback=question_data.get('feedback', '분석 중입니다.'),
            strengths=question_data.get('strengths', []),
            improvements=question_data.get('improvements', [])
        )

    def validate_detail_analysis(self, detail_analysis: Dict) -> bool:
        """상세 분석 데이터 구조 검증"""
        required_areas = ['text', 'voice', 'video', 'emotion']

        for area in required_areas:
            if area not in detail_analysis:
                raise ValueError(f"상세 분석에서 {area} 영역이 없습니다")

            area_data = detail_analysis[area]
            if 'score' not in area_data:
                raise ValueError(f"{area} 영역에 score 필드가 없습니다")

        return True


# %%
# score_aggregator

from typing import List, Dict
import statistics


class ScoreAggregator:
    """점수 집계 전용 클래스 - 가중치 적용 없이 단순 평균만 계산"""

    def aggregate_all_scores(self, question_analyses: List[QuestionAnalysis]) -> Dict:
        """
        모든 점수를 집계해서 최종 평가 점수 계산

        Args:
            question_analyses: 6개 질문의 분석 결과 (이미 가중치 적용된 점수 포함)

        Returns:
            집계된 점수 딕셔너리
        """
        # 4영역 평균 점수 계산
        area_scores = self._calculate_area_averages(question_analyses)

        # 최종 종합 점수 계산
        total_score = self._calculate_total_score(area_scores)

        # 상위 퍼센트 및 등급 계산
        rank = self._calculate_rank(total_score)
        grade = self._calculate_grade(total_score)

        return {
            'question_scores': [self._format_question_score(qa) for qa in question_analyses],
            'area_scores': area_scores,
            'total_evaluation': {
                'total_score': total_score,
                'rank': rank,
                'grade': grade
            }
        }

    def _calculate_area_averages(self, question_analyses: List[QuestionAnalysis]) -> Dict:
        """4영역(텍스트, 음성, 영상, 감정)의 평균 점수 계산"""

        # 각 영역별 점수들을 수집
        text_scores = []
        voice_scores = []
        video_scores = []
        emotion_scores = []

        # 세부 지표별 점수들도 수집
        text_similarity = []
        text_accuracy = []
        text_understanding = []

        voice_speed = []
        voice_fluency = []
        voice_tone = []

        video_gaze_rate = []
        video_shoulder_scores = []
        video_hand_scores = []

        emotion_positive = []
        emotion_neutral = []
        emotion_nervous = []
        emotion_negative = []

        # 각 질문에서 점수 추출
        for qa in question_analyses:
            detail = qa.detail_analysis

            # 텍스트 영역
            text_scores.append(detail['text']['score'])
            text_similarity.append(detail['text']['similarity'])
            text_accuracy.append(detail['text']['accuracy'])
            text_understanding.append(detail['text']['understanding'])

            # 음성 영역
            voice_scores.append(detail['voice']['score'])
            voice_speed.append(detail['voice']['speed']['score'])
            voice_fluency.append(detail['voice']['fluency']['score'])
            voice_tone.append(detail['voice']['tone']['score'])

            # 영상 영역
            video_scores.append(detail['video']['score'])
            video_gaze_rate.append(detail['video']['gaze_rate']['percentage'])
            video_shoulder_scores.append(detail['video']['shoulder_posture']['score'])
            video_hand_scores.append(detail['video']['hand_posture']['score'])

            # 감정 영역
            emotion_scores.append(detail['emotion']['score'])
            emotion_positive.append(detail['emotion']['positive'])
            emotion_neutral.append(detail['emotion']['neutral'])
            emotion_nervous.append(detail['emotion']['nervous'])
            emotion_negative.append(detail['emotion']['negative'])

        # 평균 계산 후 반환
        return {
            'text': {
                'total': round(statistics.mean(text_scores)),
                'similarity': round(statistics.mean(text_similarity)),
                'accuracy': round(statistics.mean(text_accuracy)),
                'understanding': round(statistics.mean(text_understanding))
            },
            'voice': {
                'total': round(statistics.mean(voice_scores)),
                'speed': round(statistics.mean(voice_speed)),
                'fluency': round(statistics.mean(voice_fluency)),
                'tone': round(statistics.mean(voice_tone))
            },
            'video': {
                'total': round(statistics.mean(video_scores)),
                'gaze_rate': round(statistics.mean(video_gaze_rate)),
                # 자세는 어깨 + 손 점수의 평균
                'posture': round(statistics.mean([
                    (shoulder + hand) / 2
                    for shoulder, hand in zip(video_shoulder_scores, video_hand_scores)
                ]))
            },
            'emotion': {
                'total': round(statistics.mean(emotion_scores)),
                # 감정 비율은 전체 질문의 평균
                'positive': round(statistics.mean(emotion_positive)),
                'neutral': round(statistics.mean(emotion_neutral)),
                'nervous': round(statistics.mean(emotion_nervous)),
                'negative': round(statistics.mean(emotion_negative))
            }
        }

    def _calculate_total_score(self, area_scores: Dict) -> int:
        """4영역 점수의 단순 평균으로 최종 점수 계산"""
        total = (
                        area_scores['text']['total'] +
                        area_scores['voice']['total'] +
                        area_scores['video']['total'] +
                        area_scores['emotion']['total']
                ) / 4
        return round(total)

    def _calculate_rank(self, total_score: int) -> str:
        """점수를 기반으로 상위 몇 % 계산"""
        if total_score >= 95:
            return "상위 5%"
        elif total_score >= 90:
            return "상위 10%"
        elif total_score >= 85:
            return "상위 15%"
        elif total_score >= 80:
            return "상위 25%"
        elif total_score >= 75:
            return "상위 35%"
        elif total_score >= 70:
            return "상위 50%"
        else:
            return "하위 50%"

    def _calculate_grade(self, total_score: int) -> str:
        """점수를 기반으로 등급 계산"""
        if total_score >= 95:
            return "S"
        elif total_score >= 90:
            return "A+"
        elif total_score >= 85:
            return "A"
        elif total_score >= 80:
            return "B+"
        elif total_score >= 75:
            return "B"
        elif total_score >= 70:
            return "C+"
        else:
            return "C"

    def _format_question_score(self, qa: QuestionAnalysis) -> Dict:
        """질문별 점수를 최종 형식으로 변환"""
        return {
            'question_id': qa.question_id,
            'name': qa.name,
            'type': qa.type,
            'score': qa.final_score,  # 이미 가중치가 적용된 최종 점수
            'question': qa.question,
            'my_answer': qa.my_answer,
            'model_answer': qa.model_answer,
            'detail_analysis': qa.detail_analysis,
            'feedback': qa.feedback,
            'strengths': qa.strengths,
            'improvements': qa.improvements
        }


# %%
class GeminiAdvisor:
    """Gemini AI를 활용한 AI 조언 생성 클래스 - 병렬 처리 최적화"""

    def __init__(self, model, generation_config, safety_settings):
        self.model = model
        self.generation_config = generation_config
        self.safety_settings = safety_settings

    def generate_all_advice(
            self,
            question_analyses: List["QuestionAnalysis"],
            aggregated_scores: Dict,
            user_nickname: str
    ) -> Dict:
        # 공통 데이터 전처리 (한 번만 계산)
        analysis_summary = self._prepare_analysis_summary(question_analyses)
        question_summary = self._prepare_question_summary(question_analyses)

        # 병렬 처리로 4개 API 호출 동시 실행
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 4개 작업을 동시에 제출
            future_personalized = executor.submit(
                self._generate_personalized_message_with_summary,
                question_summary, aggregated_scores['total_evaluation'], user_nickname
            )
            future_strengths = executor.submit(
                self._generate_top_strengths_with_summary,
                analysis_summary, user_nickname
            )
            future_improvements = executor.submit(
                self._generate_improvements_with_summary,
                analysis_summary, user_nickname
            )
            future_summaries = executor.submit(
                self._generate_question_summaries,
                question_analyses
            )

            # 모든 결과를 기다림 (가장 오래 걸리는 작업 시간만큼만 소요)
            personalized_message = future_personalized.result()
            top_strengths = future_strengths.result()
            improvements = future_improvements.result()
            question_summaries = future_summaries.result()

        return {
            'personalized_message': personalized_message,
            'top_strengths': top_strengths,
            'improvements': improvements,
            'question_summaries': question_summaries
        }

    def _prepare_analysis_summary(self, question_analyses: List["QuestionAnalysis"]) -> str:
        """강점/개선점 분석용 공통 요약 생성 (한 번만 계산)"""
        return "\n".join([
            f"{qa.name} ({qa.type}, {qa.final_score}점):\n"
            f"- 텍스트: {qa.detail_analysis['text']['score']}점\n"
            f"- 음성: {qa.detail_analysis['voice']['score']}점\n"
            f"- 영상: {qa.detail_analysis['video']['score']}점\n"
            f"- 감정: {qa.detail_analysis['emotion']['score']}점\n"
            f"- 강점: {', '.join(qa.strengths)}\n"
            f"- 개선점: {', '.join(qa.improvements)}"
            for qa in question_analyses
        ])

    def _prepare_question_summary(self, question_analyses: List["QuestionAnalysis"]) -> str:
        """개인 맞춤 조언용 질문 요약 생성 (한 번만 계산)"""
        return "\n".join([
            f"- {qa.name} ({qa.type}, {qa.final_score}점): {qa.feedback}\n"
            f"  강점: {', '.join(qa.strengths)}\n"
            f"  개선점: {', '.join(qa.improvements)}"
            for qa in question_analyses
        ])

    def _generate_personalized_message_with_summary(
            self,
            question_summary: str,
            total_evaluation: Dict,
            user_nickname: str
    ) -> str:
        prompt = f"""
{user_nickname}님의 면접 전체를 분석해서 개인 맞춤 조언을 생성해주세요.

전체 점수: {total_evaluation['total_score']}점 ({total_evaluation['rank']})

각 질문별 분석:
{question_summary}

조언 생성 요구사항:
- 정확히 200자 내외
- {user_nickname}님의 데이터에서만 발견되는 구체적 특징 기반
- 가장 시급한 개선점 1개 + 활용할 최고 강점 1개 명시
- 실행 가능한 구체적 액션 아이템 포함
- 따뜻하되 예리한 분석이 느껴지는 전문적 톤
"""
        return self._call_gemini(prompt).strip()

    def _generate_top_strengths_with_summary(self, analysis_summary: str, user_nickname: str) -> List[Dict]:
        prompt = f"""
{user_nickname}님의 6개 질문 면접 분석 결과를 종합해서 TOP 3 강점을 선정해주세요.

강점top3:
{analysis_summary}

요구사항:
1. 전체적으로 가장 뛰어난 TOP 3 강점 선정
2. 중복 제거 및 개인 특성 반영
3. 각각 제목(간결) + 설명(구체적) + 점수
4. 반드시 20자 내외로 간결하고 직관적이게 작성

⚠️ 반드시 다음 JSON 형태로만 출력하세요. 설명 문구 없이 JSON 객체만 출력해주세요.

[
  {{"title": "강점 제목", "description": "상세 설명", "score": 94}},
  {{"title": "강점 제목", "description": "상세 설명", "score": 91}},
  {{"title": "강점 제목", "description": "상세 설명", "score": 89}}
]
"""
        response = self._call_gemini(prompt).strip()
        return self._parse_json_response(response, "강점")

    def _generate_improvements_with_summary(self, analysis_summary: str, user_nickname: str) -> List[Dict]:
        prompt = f"""
{user_nickname}님의 6개 질문 면접 분석 결과를 종합해서 우선순위별 TOP 3 개선점을 선정해주세요.

개선점top3:
{analysis_summary}

요구사항:
1. 우선순위별 TOP 3 개선점 (1순위가 가장 중요)
2. 중복 제거 및 개인화
3. 각각 제목(간결) + 설명(구체적) + 점수
4. 반드시 15자 내외의 한문장으로 작성
5. 간결하고 직관적인 표현의 한문장

⚠️ 반드시 다음 JSON 형태로만 출력하세요. 설명 문구 없이 JSON 객체만 출력해주세요.

[
  {{"priority": 1, "title": "개선점 제목", "description": "상세 설명", "score": 78}},
  {{"priority": 2, "title": "개선점 제목", "description": "상세 설명", "score": 82}},
  {{"priority": 3, "title": "개선점 제목", "description": "상세 설명", "score": 85}}
]
"""
        response = self._call_gemini(prompt).strip()
        return self._parse_json_response(response, "개선점")

    def _generate_question_summaries(self, question_analyses: List["QuestionAnalysis"]) -> List[str]:
        # 질문 요약은 각각 독립적이므로 병렬 처리
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for qa in question_analyses:
                prompt = f"""
    면접 답변을 한 줄로 요약해주세요.

    질문: {qa.question}
    답변: {qa.my_answer}
    점수: {qa.final_score}점
    강점: {', '.join(qa.strengths)}

    요구사항:
    - 정확히 15자 내외
    - 답변의 핵심 강점을 표현
    - 완성된 문장으로 작성
    - "~했습니다" 또는 "~되었습니다" 형태

    예시:
    - "개념 설명이 명확하고 체계적이었습니다"
    - "기술적 이해도가 뛰어나고 정확했습니다"
    - "문제해결 과정이 논리적이었습니다"
    - "실무 경험이 잘 반영된 답변이었습니다"
    - "창의적 사고가 돋보이는 답변이었습니다"

    위 형태로 한 문장만 출력하세요:
    """
                futures.append(executor.submit(self._call_gemini, prompt))

            # 순서대로 결과 수집
            return [future.result().strip() for future in futures]

    def _parse_json_response(self, response: str, error_type: str) -> List[Dict]:
        """JSON 파싱 통합 처리"""
        response = self._clean_json_block(response)
        try:
            parsed = json.loads(response)
            print(f"✅ JSON 형식 파싱 성공! ({error_type})")
            return parsed
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패 ({error_type}):", str(e))
            print("========== Gemini 원본 응답 ==========")
            print(response)
            print("===================================")
            raise Exception(f"Gemini 응답이 JSON 형식이 아닙니다. ({error_type})")

    def _call_gemini(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            if not response.candidates:
                raise Exception("Gemini 응답이 비어있습니다.")

            candidate = response.candidates[0]
            if candidate.finish_reason.name != "STOP":
                raise Exception(f"Gemini 응답이 정상 종료되지 않았습니다. finish_reason: {candidate.finish_reason.name}")

            if not candidate.content.parts:
                raise Exception("Gemini 응답이 Part를 포함하지 않습니다.")

            return candidate.content.parts[0].text.strip()

        except Exception as e:
            raise Exception(f"Gemini API 호출 실패: {str(e)}")

    def _clean_json_block(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```")
        return text.strip().strip("`").strip()


# %%
# report_builder

from typing import Dict, List
from datetime import datetime


class ReportBuilder:
    """최종 보고서 구성 클래스"""

    def get_grade_message(self, score: int) -> str:
        """점수별 메시지 생성 - 프론트엔드와 동일한 로직"""
        if score >= 95:
            return "S등급 - 면접관도 감탄할만큼 완벽한 응답이네요!"
        elif score >= 90:
            return "A+등급 - 완벽에 가까운 훌륭한 답변입니다!"
        elif score >= 85:
            return "A등급 - 우수한 답변이에요! 약간만 다듬으면 더 좋아질 수 있어요."
        elif score >= 80:
            return "B+등급 - 매일 연습하면 더좋은 답변을 할 수 있어요!"
        elif score >= 75:
            return "B등급 - 기본기는 탄탄해요! 조금 더 연습해보세요."
        elif score >= 70:
            return "C+등급 - 시작이 반이에요. 한걸음씩 함께 노력해요!"
        else:
            return "C등급 - 꾸준한 연습으로 실력을 키워나가세요!"

    def build_final_report(
            self,
            user_info: UserInfo,
            question_analyses: List[QuestionAnalysis],
            aggregated_scores: Dict,
            ai_advice: Dict,
            step_names: List[str]

    ) -> Dict:
        """
        최종 보고서 데이터를 프론트엔드 형식에 맞게 구성

        Args:
            user_info: 사용자 정보
            question_analyses: 질문별 분석 결과
            aggregated_scores: 집계된 점수
            ai_advice: LLM 생성 조언
            step_names: 단계명 리스트

        Returns:
            최종 보고서 딕셔너리
        """
        grade_message = self.get_grade_message(aggregated_scores['total_evaluation']['total_score'])

        # 질문별 데이터에 LLM 생성 요약 추가
        enhanced_question_scores = []
        for i, qa_dict in enumerate(aggregated_scores['question_scores']):
            enhanced_qa = qa_dict.copy()
            enhanced_qa['summary'] = ai_advice['question_summaries'][i]  # 15자 내외 요약 추가
            enhanced_question_scores.append(enhanced_qa)

        # 최종 보고서 구성
        final_report = {
            # 사용자 기본 정보
            'user_info': {
                'user_id': user_info.user_id,
                'user_nickname': user_info.user_nickname,
                'interview_id': user_info.interview_id,
                'interview_date': user_info.interview_date,
                'interview_duration': user_info.interview_duration
            },

            # 종합 평가 점수
            'total_evaluation': aggregated_scores['total_evaluation'],

            # 등급 메시지
            'total_evaluation': {
                **aggregated_scores['total_evaluation'],
                'grade_message': grade_message
            },

            # 4영역 평균 점수들
            'area_scores': aggregated_scores['area_scores'],

            # 6개 질문별 상세 점수들 (요약 포함)
            'question_scores': enhanced_question_scores,

            # LLM이 생성한 AI 조언들
            'ai_advice': {
                'personalized_message': ai_advice['personalized_message'],
                'top_strengths': ai_advice['top_strengths'],
                'improvements': ai_advice['improvements']
            }
        }

        return final_report


# %%
# 최종 보고서 생성 클래스
class FinalEvaluationGenerator:
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=10000,
        )

        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]

        self.step_names = [
            '아이스브레이킹', '질문 1', '질문 2', '질문 3',
            '질문 4', '질문 5', '질문 6', '최종 평가'
        ]

    def generate_final_report(self, user_info: UserInfo, question_analyses: List[QuestionAnalysis]) -> Dict:
        try:
            # 1. 점수 집계
            score_aggregator = ScoreAggregator()
            aggregated_scores = score_aggregator.aggregate_all_scores(question_analyses)

            # 2. AI 조언 생성
            gemini_advisor = GeminiAdvisor(self.model, self.generation_config, self.safety_settings)
            ai_advice = gemini_advisor.generate_all_advice(
                question_analyses,
                aggregated_scores,
                user_info.user_nickname
            )

            # 3. 보고서 구성
            report_builder = ReportBuilder()
            final_report = report_builder.build_final_report(
                user_info,
                question_analyses,
                aggregated_scores,
                ai_advice,
                self.step_names
            )

            return final_report

        except Exception as e:
            print(f"[오류] 최종 보고서 생성 실패: {str(e)}")
            raise

    def generate_final_report_from_json(self, interview_data) -> Dict:
        try:
            # 1. 데이터 파싱
            data_parser = DataParser()
            parsed_data = data_parser.parse_interview_data(interview_data)
            user_info = parsed_data['user_info']
            question_analyses = parsed_data['question_analyses']

            # 2. 보고서 생성
            return self.generate_final_report(user_info, question_analyses)

        except Exception as e:
            print(f"[오류] JSON 데이터 처리 실패: {str(e)}")
            raise





