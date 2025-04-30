# invoice_extraction_chain.py

import logging
import json
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from fastapi import FastAPI


OLLAMA_URL = "http://localhost:6006"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TEMP = float(os.getenv("LLM_TEMP", "0.1"))


def get_template(output_ocr: str) -> ChatPromptTemplate:
    """
    Return the full prompt for extracting invoice fields from OCR output.
    `input_message` should be the placeholder {ocr_output} when used with PromptTemplate.
    """

    SYSTEM_MSG = """ You are a strict information-extraction engine.
     Your sole task is to read the plain-text of one invoice (no structured markup)
     and return a single JSON object that matches the schema below exactly—nothing more, nothing less.
     {{
      "name": null,
      "address": null,
      "company_taxpayer_id": null,
      "total_including_taxes": null,
      "total_excluding_taxes": null,
      "tax_amount": null,
      "tax_rate": null,
      "invoice_number": null,
      "purchase_order": null,
      "iban": null,
      "currency": null,
      "language": null,
      "invoice_date": null,
      "due_date": null
     }}

     Extraction rules:
     1. Absolute accuracy is critical. If you are not 100 % certain, output null.
     2. Keep the field order unchanged. Output a single JSON object, no extra keys.
     3. Normalize numbers to two decimals, dot‐separator.
     4. Normalize dates to YYYY-MM-DD.
     5. Validate formats (currency, language, IBAN).
     6. No partial strings; extract full values.
     7. If a field appears multiple times, pick the authoritative one.
     8. Never invent data.
     9. If multiple invoices appear, return "ERROR: multiple invoices detected".
     10. Respond with UTF-8 JSON only.
    """

    #     SYSTEM_INSTRUCTIONS = """
    #  You are a strict information-extraction engine.
    #  Your sole task is to read the plain-text of one invoice (no structured markup)
    #  and return a single JSON object that matches the schema below exactly—nothing more, nothing less.
    #  {{
    #   "name": null,
    #   "address": null,
    #   "company_taxpayer_id": null,
    #   "total_including_taxes": null,
    #   "total_excluding_taxes": null,
    #   "tax_amount": null,
    #   "tax_rate": null,
    #   "invoice_number": null,
    #   "purchase_order": null,
    #   "iban": null,
    #   "currency": null,
    #   "language": null,
    #   "invoice_date": null,
    #   "due_date": null
    #  }}

    #  Extraction rules:
    #  1. Absolute accuracy is mandatory. If you are not 100 % certain, output null.
    #  2. Keep the field order unchanged. Output a single JSON object, no extra keys.
    #  3. Normalize numbers to two decimals, dot‐separator.
    #  4. Normalize dates to YYYY-MM-DD.
    #  5. Validate formats (currency, language, IBAN).
    #  6. No partial strings; extract full values.
    #  7. If a field appears multiple times, pick the authoritative one.
    #  8. Never invent data.
    #  9. If multiple invoices appear, return "ERROR: multiple invoices detected".
    #  10. Respond with UTF-8 JSON only.
    # """

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
    # prompt = PromptTemplate(template=template, input_variables=["ocr_output"])

    # optional in‑memory chat history

    return LLMChain(llm=llm, prompt=prompt, verbose=False)


app = FastAPI()
chain = build_chain()


@app.post("/extract")
async def extract(invoice_text: str):
    """
    FastAPI endpoint to extract invoice data from OCR text.
    """
    # run the chain and return the result
    result_json = await chain.run({"ocr_output": invoice_text})
    return result_json


if __name__ == "__main__":
    # example OCR output
    batches = [
        """
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
                """,
        """Invoice
THE OFFICE SOLUTIONS ORGANISATION
J.Black&Partners
9Shakespeare Road
Wellesbourne
Stratford
warwickshire
15
WK43VG
Page 1of1.
InvoiceAddress:
DeliveryAddress:
DEMO Co.LTD
DEMO Co.LTD
5 Priory Mews
5Priory Mews
Monks Ferry
Monks Ferry
Birkenhead
Birkenhead
Wirral
Wirral
CH41 5AZ
CH41 5AZ
DM/0
Delivery Ref.
YourOrder/Reference
Date
Invoice Number
Account
D16352
05-Jan-04
15/112171
BLA001
Product
Description
Price
Per
Qty
Disc.
NetValue
VAT
Ref:D16352004
7132
Sasco 2400188ar
8.93
1
8.93""",
        """
Invoice Number:
1-996-84199
Invoice Date:
Sep01.2014
Account Number:
1334-8037-4
Page:
1of2
FedExTaxID:71-0427007
IRISINC
RECEIVED
SHARON ANDERSON
4731WATLANTICAVESTEB1
DELRAYBEACHFL33445-3897
SEP
RECU
Invoice Questions?
BillingAccount Shipping Address:
BY:
ContactFedEx Revenue Services
IRISINC
4731WATLANTICAVE
Phone:
(800)622-1147M-F7-6(CST)
DELRAY BEACHFL33445-3897 US
Fax:
(800)548-3020
Internet:
www.fedex.com
InvoiceSummarySep01.2014
FedEx GroundServices
Other Charges
11.00
Total Charges
USDS
11.00
TOTALTHISINVOICE
USDS
11.00
The only charges accrued for this period is the Weekly Service Charge.
The FedExGround accounts eletencedin thisivoice have been trnsterredandassignedtoare owned by.andre payable to FedEx Express
Toensureproporcredit.pleasereturnthisportionwithyourpaymenttoFedEx.
Please donot staple orfold.Plonse make yourcheckpayable toFedEx.
For chge of adssck hnd co lm oe i
Invoice
Account
Amount
Remittance Advice
Number
Number
Due
Yourpaymentis due by Sep16.2004
1-996-84199
1334-8037-4
USDS11.00
133480371996841993200000110071
AT01031292 46844B196A**3DGT
IRISINC
SHARONANDERSON
4731WATLANTICAVESTEB1
FedEx
DELRAYBEACHFL33445-3897
P.O.Box94515
PALATINEIL60094-4515
""",
        """
King.Richards and Roy
Invoicenumber INV/69-33/009
Email:smithjeffrey@example.net
Note: All payments to be made in cash.
Buyer:AndrewGreen
Contact us for queries on these quotations.
8274BradshawPath
NewMariaborough,MT23884US
Tel:+(825)267-2334
Emai:ndonaldson@example.net
Invoice Date:21-Mar-2020
Site:https://flores.com/
Item
Price
Quantity
Amount
Land response different.
62.41
1.00
62.41
Writer difference.
31.88
5.00
159.40
College.
0.39
2.00
0.78
Own eatleast
21.11
6.00
126.66
TOTAL:367.25EUR
Total in words:three hundred and sixt-
y-seven point two five
""",
    ]

    # run the chain and print the extracted JSON
    batch_inputs = [{"ocr_output": ocr_text} for ocr_text in batches]

    # Test using chain.batch
    results = chain.batch(batches)

    # Print results for all batches
    parsed = []
    for item in results:
        # item["text"] is a JSON‐string; load it into a dict
        try:
            obj = json.loads(item["text"])
        except json.JSONDecodeError:
            # handle or skip bad JSON
            continue
        parsed.append(obj)

    # 3. Write the clean array to disk
    with open("extracted_invoices_clean.json", "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"Wrote {len(parsed)} invoice records to extracted_invoices_clean.json")


# app = FastAPI()


# @app.post("/extract")
# async def extract(invoice_text: str):
#     return await chain.ainvoke({"ocr_output": invoice_text})
