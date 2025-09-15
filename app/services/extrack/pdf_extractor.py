import pdfplumber

def extract_text_from_pdf(file_obj) -> str:
    full_text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text.strip()
