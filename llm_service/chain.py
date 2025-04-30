# invoice_extraction_chain.py

import logging
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os


OLLAMA_URL = "http://localhost:6006"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TEMP = float(os.getenv("LLM_TEMP", "0.1"))


def get_template(output_ocr: str) -> ChatPromptTemplate:
    """
    Return the full prompt for extracting invoice fields from OCR output.
    `input_message` should be the placeholder {ocr_output} when used with PromptTemplate.
    """

    SYSTEM_MSG = """Extract invoice data from OCR’d text into this exact JSON schema.  

    **Output ONLY valid JSON**, nothing else.

    ### Schema:
    {{
    "invoice_number": string,
    "seller": {{
        "name": string,
        "address": string,
        "country": string
    }},
    "invoice_date": string,    // DD/MM/YYYY
    "due_date": string,        // DD/MM/YYYY
    "client": {{
        "name": string,
        "address": string,
        "reference": string,     // phone or other ref
        "country": string
    }},
    "items": [
        {{

        "description": string,
        "amount": number,
        "vat_amount": number,
        "vat_rate": string

        }}
    ],
    "total": number,
    "total_vat": number,
    "total_due": number,
    "issued_by": string

    }} 
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_MSG),
            HumanMessagePromptTemplate.from_template(
                "Input invoice text:\n{ocr_output}"
            ),
        ]
    )
    return prompt


def build_chain() -> LLMChain:
    # enable INFO‑level logging from langchain
    logging.basicConfig(level=logging.INFO)
    llm = Ollama(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMP,
        keep_alive=-1,
        format="json",
        num_ctx=8192,
        cache=False,
    )

    # build a PromptTemplate that injects the OCR output at runtime
    prompt = get_template("{ocr_output}")

    return LLMChain(llm=llm, prompt=prompt, verbose=False)


chain = build_chain()


if __name__ == "__main__":
    # example OCR output
    ocr_text = """
                77 Hammersmith Road
                West Kensington
                London, W14 0QH
                Phone: 0208 668 381
                Invoice no.:1
                Invoice Date:31/08/2020
                Buyer Ltd.
                Payment terms:30 days
                Billy Buyer
                Due date30/09/2020
                43 Customer Road
                Manchester, M4 1HS
                United Kingdom
                Add any additional instructions or terms here.
                Description
                Date
                Qty
                Unit Price
                VAT%
                Total
                Client work
                31/08/2020
                3
                60,00 GBP
                20 %
                180,00 GBP
                Product A
                31/08/2020
                10
                14,00 GBP
                20 %
                140,00 GBP
                Product B
                31/08/2020
                2
                12,00 GBP
                20 %
                24,00 GBP
                Net total
                344,00 GBP
                VAT 20%
                68,80 GBP
                Total amount due
                412,80 GBP
                Your Company Name
                Contact Information
                Payment Details
                77 Hammersmith Road
                Freddy Seller
                Bank Name
                Barclays PLC
                West Kensington
                Phone: 0208 668 381
                Sort-Code
                20-84-12
                London, W14 0QH
                Email: freddy@mycompany.co.uk
                Account No.
                12345678
                VAT No. GB123 4567 89
                www.mycompany.co.uk
                """

    # run the chain and print the extracted JSON
    result_json = chain.run(ocr_output=ocr_text)
    print(result_json)
