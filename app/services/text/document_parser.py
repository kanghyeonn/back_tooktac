from google.cloud import documentai
from google.cloud import storage
from prettytable import PrettyTable
import google.generativeai as genai
import re
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from app.services.extrack.hwpx_extractor import hwpx_to_html
from app.services.extrack.docx_extractor import extract_text_from_docx


from dotenv import load_dotenv
import os

load_dotenv()


gemini_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_api_key)

project_id = os.getenv("PROJECT_ID")
processor_id = os.getenv("PROCESSOR_ID")
location = os.getenv("LOCATION", "us")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

class VertexDocumentParser:
    """Vertex AI Document AI와 Gemini를 사용한 문서 파싱 및 구조화 클래스"""

    def __init__(self, project_id: str, location: str, processor_id: str, gemini_api_key: str):
        self.processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        self.client = documentai.DocumentProcessorServiceClient()
        self.storage_path = "parsed_documents/"

        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

        os.makedirs(self.storage_path, exist_ok=True)

    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        다양한 형식의 문서를 파싱하고 구조화된 정보를 반환
        지원 형식: PDF, HWPX, DOCX
        """
        file_path = Path(file_path)
        text = ""

        # 1. PDF or 이미지 파일 → Document AI 사용
        if file_path.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']:
            with open(file_path, "rb") as f:
                file_content = f.read()

            mime_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else "image/jpeg"

            result = self.client.process_document(
                request=documentai.ProcessRequest(
                    name=self.processor_name,
                    raw_document=documentai.RawDocument(
                        content=file_content,
                        mime_type=mime_type
                    )
                )
            )
            text = result.document.text
            page_count = len(result.document.pages)

        # 2. HWPX 파일
        elif file_path.suffix.lower() == ".hwpx":
            text = hwpx_to_html(str(file_path))
            page_count = 1  # 추정값, 필요 시 보완

        # 3. DOCX 파일
        elif file_path.suffix.lower() == ".docx":
            text = extract_text_from_docx(str(file_path))
            page_count = 1

        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_path.suffix}")

        # 4. Gemini로 구조화
        structured_data = self._structure_with_gemini(text)

        return {
            "full_text": text,
            "pages": page_count,
            "structured_content": structured_data
        }

    def _structure_with_gemini(self, text: str) -> Dict[str, Any]:
        """Gemini를 사용해서 면접 질문 생성에 필요한 정보만 자동 구조화"""
        prompt = f"""
        다음 이력서/자기소개서 텍스트를 분석해서 면접 질문 생성에 필요한 정보만 JSON 형식으로 구조화해주세요.
        개인 식별 정보(이름, 전화번호, 이메일, 주소)는 제외하고 추출하세요.

        텍스트:
        {text}

        다음 형식으로 추출해주세요:
        {{
            "education": {{
                "school": "학교명",
                "major": "전공",
                "period": "재학기간",
                "gpa": "학점"
            }},
            "skills": ["기술1", "기술2", ...],
            "certifications": ["자격증1", "자격증2", ...],
            "career": {{
                "status": "신입/경력",
                "experience": "경력 상세 (있는 경우)",
                "years": "경력 년수"
            }},
            "projects": [
                {{
                    "name": "프로젝트명",
                    "description": "설명",
                    "tech_stack": ["사용 기술"],
                    "period": "기간"
                }}
            ],
            "desired_position": {{
                "location": "희망근무지",
                "salary": "희망연봉",
                "job_type": "희망 직무",
                "industry": "희망 산업분야"
            }},
            "self_introduction": {{
                "motivation": "지원동기",
                "strengths": "강점",
                "career_goals": "목표",
                "key_experiences": "주요 경험"
            }}
        }}

        JSON만 출력하고 다른 설명은 하지 마세요.
        정보가 없는 필드는 null 또는 빈 값으로 처리하세요.
        """

        response = self.gemini_model.generate_content(prompt)

        try:
            cleaned = self._strip_code_block(response.text)
            return json.loads(cleaned)
        except Exception as e:
            print("❌ Gemini 응답을 JSON으로 파싱하지 못했습니다.")
            print("📄 원본 응답:\n", response.text)
            raise e

    @staticmethod
    def _strip_code_block(text: str) -> str:
        """Gemini가 ```json ... ``` 형태로 감싼 응답 제거"""
        return re.sub(r"^```(?:json)?\n?|```$", "", text.strip(), flags=re.MULTILINE)

    def save_parsed_data(self, file_path: str, parsed_data: Dict[str, Any]) -> str:
        """파싱된 데이터를 저장하고 문서 ID 반환"""
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]

        save_path = os.path.join(self.storage_path, f"{doc_id}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "doc_id": doc_id,
                "original_file": os.path.basename(file_path),
                "parsed_at": datetime.now().isoformat(),
                "data": parsed_data
            }, f, ensure_ascii=False, indent=2)

        return doc_id

    def parse_and_save(self, file_path: str) -> str:
        """문서를 파싱하고 저장하는 통합 메서드"""
        parsed = self.parse_document(file_path)
        return self.save_parsed_data(file_path, parsed)
