from fastapi import UploadFile
from .docx_extractor import extract_text_from_docx
from .pdf_extractor import extract_text_from_pdf
from .hwpx_extractor import hwpx_to_html

def extract_resume_text(file: UploadFile) -> str:
    if file.filename.endswith(".docx"):
        return extract_text_from_docx(file)
    elif file.filename.endswith(".pdf"):
        return extract_text_from_pdf(file.file)
    elif file.filename.endswith(".hwpx"):
        return hwpx_to_html(file.file)
    else:
        return ""
