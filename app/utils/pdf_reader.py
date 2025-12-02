import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF using pdfplumber.
    """
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"

    return full_text
