from fastapi import UploadFile
from docx import Document
import os

def extract_text_from_docx(uploaded_file: UploadFile) -> str:
    contents = uploaded_file.file.read()
    with open("temp.docx", "wb") as f:
        f.write(contents)
    doc = Document("temp.docx")
    os.remove("temp.docx")
    return "\n".join(p.text for p in doc.paragraphs)
