class GetTemplate:
    """Class to build a combined system prompt with OCR text."""

    def __init__(self, ocr_text: str):
        self.ocr_text = ocr_text

    def generate_prompt(self) -> str:
        """Generate the complete prompt including instructions and OCR content."""
        system_instruction = (
            "Extract all relevant invoice details from the text and format them into a JSON object."
            " Include fields such as invoice number, date, seller and buyer information, itemized list, totals, etc."
            " If a field is not present, omit it or set it to null and DO NOT HALLUCINATE."
        )
        return f"{system_instruction}\n\n{self.ocr_text}"
