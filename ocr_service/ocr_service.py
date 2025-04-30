from fastapi import FastAPI, UploadFile, File
from ocr_helper import extract_text_from_image

app = FastAPI()


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    # save to temp, run extract_text_from_image, return plain text
    import tempfile

    suffix = file.filename and file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix="." + suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    text = extract_text_from_image(path)
    return {"text": text}
