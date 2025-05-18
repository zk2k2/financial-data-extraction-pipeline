from pathlib import Path
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

ingine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def extract_text_from_image(file_path: str) -> str:
    """
    Runs OCR on a given file (image or PDF) and returns the extracted plain text.

    - For images (.png, .jpg, .jpeg, .tiff), runs OCR directly on the file.
    - For PDFs, converts each page to an image and runs OCR on each page.

    Returns:
        A single string containing all detected lines separated by newlines.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    lines = []

    if suffix == ".pdf":
        # Convert PDF pages to images
        pages = convert_from_path(file_path)
        for page in pages:
            # OCR on PIL Image
            result = ingine.ocr(page, cls=True)
            for page_res in result:
                for line in page_res:
                    lines.append(line[1][0])
    else:
        # Assume it's an image file
        result = ingine.ocr(str(path), cls=True)
        for page_res in result:
            for line in page_res:
                lines.append(line[1][0])

    return "\n".join(lines)
