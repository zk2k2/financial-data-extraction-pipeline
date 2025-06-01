from pathlib import Path
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

ingine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)



def extract_text_from_image_gpu(file_path: str) -> str:
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


def extract_text_from_image(path):
    # Initialize PaddleOCR with CPU usage
    ocr_engine = PaddleOCR(
        use_angle_cls=True, 
        lang='en',
        use_gpu=False,  # Force CPU usage
        show_log=False
    )
    
    result = ocr_engine.ocr(str(path), cls=True)
    
    # Extract text from OCR results
    extracted_text = ""
    if result and result[0]:
        for line in result[0]:
            if len(line) > 1:
                extracted_text += line[1][0] + "\n"
    print("Extracted text:", extracted_text.strip())
    
    return extracted_text.strip()