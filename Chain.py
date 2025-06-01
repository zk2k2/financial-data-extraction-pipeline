import os
import json
from dotenv import load_dotenv
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.LLMEnums import OpenAIEnums
from helpers.config import get_settings

# Load environment variables
load_dotenv()

# Create settings object with required configuration
config = get_settings()
if not config:
    raise ValueError(
        "Failed to load configuration. Check your .env file or environment variables."
    )

# Check if OpenAI API key is set
if not config.OPENAI_API_KEY:
    raise ValueError(
        "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
    )
# Create the OpenAI provider using the factory
factory = LLMProviderFactory(config)
llm_provider = factory.create("openai")
if not llm_provider:
    raise ValueError("Failed to create LLM provider. Check your configuration.")
model_id = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
llm_provider.set_generation_model(model_id)

# Define the system prompt template (same as in Chain.py)
SYSTEM_MSG = """Extract invoice data from OCR'd text into this exact JSON schema.  

**Output ONLY valid JSON**, nothing else.

### Schema:
{
  "supplier": {
    "name": string,
    "address": string,
    "email": string,
    "phone_number": string,
    "vat_number": string,
    "website": string
  },
  "invoice_date": string,       // DD/MM/YYYY
  "invoice_number": string,
  "currency": string,
  "total_net": number,
  "total_tax": number,
  "total_amount_incl_tax": number
}
"""



def extract_invoice_data(ocr_text):
    """
    Extract structured invoice data from OCR text using OpenAI provider
    """
    # Construct chat history with system and user messages
    if not llm_provider:
        raise ValueError("LLM provider is not initialized.")
    chat_history = [
        llm_provider.construct_prompt(SYSTEM_MSG, OpenAIEnums.SYSTEM.value),
        llm_provider.construct_prompt(
            f"Input invoice text:\n{ocr_text}", OpenAIEnums.USER.value
        ),
    ]

    # Generate response
    response = llm_provider.generate_text(
        prompt="", chat_history=chat_history, temperature=0.1
    )
    # Check if the response is empty
    if not response:
        return {"error": "Empty response from LLM"}

    # Try to parse the response as JSON
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # If the response isn't valid JSON, return the raw text
        return {"error": "Invalid JSON response", "raw_text": response}


if __name__ == "__main__":
    # Example OCR text (same as in Chain.py)
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

    # Extract and print the structured data
    result = extract_invoice_data(ocr_text)
    print(json.dumps(result, indent=4))
