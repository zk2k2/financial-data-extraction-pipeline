import re
from datetime import datetime
from typing import Dict, Any, List, Tuple
from decimal import Decimal, InvalidOperation

class InvoiceDataValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_invoice(self, invoice_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """
        Validate and clean invoice data with field mapping
        Returns: (cleaned_data, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # First, normalize the data structure
        normalized_data = self._normalize_llm_output(invoice_data)
        
        cleaned_data = {}
        
        # Validate supplier information
        cleaned_data.update(self._validate_supplier_info(normalized_data))
        
        # Validate financial data
        cleaned_data.update(self._validate_financial_data(normalized_data))
        
        # Validate dates
        cleaned_data.update(self._validate_dates(normalized_data))
        
        # Validate contact information
        cleaned_data.update(self._validate_contact_info(normalized_data))
        
        # Cross-validate financial calculations
        self._cross_validate_amounts(cleaned_data)
        
        return cleaned_data, self.errors, self.warnings
    
    def _normalize_llm_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize nested LLM output to flat structure expected by validator"""
        normalized = {}
        
        # Handle nested supplier data
        if 'supplier' in data and isinstance(data['supplier'], dict):
            supplier = data['supplier']
            normalized['supplier_name'] = supplier.get('name', '')
            normalized['supplier_address'] = supplier.get('address', '')
            normalized['supplier_email'] = supplier.get('email', '')
            normalized['supplier_phone_number'] = supplier.get('phone_number', '')
            normalized['supplier_vat_number'] = supplier.get('vat_number', '')
            normalized['supplier_website'] = supplier.get('website', '')
        else:
            # Handle flat structure (fallback)
            normalized['supplier_name'] = data.get('supplier_name', '')
            normalized['supplier_address'] = data.get('supplier_address', '')
            normalized['supplier_email'] = data.get('supplier_email', '')
            normalized['supplier_phone_number'] = data.get('supplier_phone_number', '')
            normalized['supplier_vat_number'] = data.get('supplier_vat_number', '')
            normalized['supplier_website'] = data.get('supplier_website', '')
        
        # Handle date fields with different possible names
        normalized['expense_date'] = (
            data.get('expense_date') or 
            data.get('invoice_date') or 
            data.get('date') or
            ''
        )
        
        # Handle invoice number
        normalized['invoice_number'] = data.get('invoice_number', '')
        
        # Handle currency
        normalized['currency'] = data.get('currency', '')
        
        # Handle financial fields with different possible names
        normalized['total_net'] = (
            data.get('total_net') or 
            data.get('subtotal') or 
            data.get('net_amount') or
            None
        )
        
        normalized['total_tax'] = (
            data.get('total_tax') or 
            data.get('tax_amount') or 
            data.get('vat_amount') or
            None
        )
        
        normalized['total_amount'] = (
            data.get('total_amount') or 
            data.get('total_amount_incl_tax') or 
            data.get('grand_total') or
            data.get('amount_due') or
            None
        )
        
        return normalized
    
    def _validate_supplier_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate supplier information - make more lenient"""
        result = {}
        
        # Supplier name - be more lenient
        supplier_name = self._clean_string(data.get("supplier_name"))
        if supplier_name and supplier_name.strip():
            if len(supplier_name) > 255:
                self.warnings.append("Supplier name truncated to 255 characters")
                supplier_name = supplier_name[:255]
            result["supplier_name"] = supplier_name
        else:
            # Don't make this a hard error - add warning instead
            self.warnings.append("Supplier name is missing or empty")
        
        # VAT number validation - more lenient
        vat_number = self._clean_string(data.get("supplier_vat_number"))
        if vat_number and vat_number.strip():
            result["supplier_vat_number"] = vat_number
        
        # Address
        address = self._clean_string(data.get("supplier_address"))
        if address and address.strip():
            result["supplier_address"] = address
        
        # Website
        website = self._clean_string(data.get("supplier_website"))
        if website and website.strip():
            if not website.startswith(('http://', 'https://')):
                website = 'https://' + website
            result["supplier_website"] = website
        
        return result
    
    def _validate_financial_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate financial amounts"""
        result = {}
        
        # Currency
        currency = self._clean_string(data.get("currency"))
        if currency:
            currency = currency.upper()
            if len(currency) > 10:
                self.warnings.append("Currency code truncated")
                currency = currency[:10]
            result["currency"] = currency
        
        # Validate amounts
        amounts = ["total_net", "total_tax", "total_amount"]
        for amount_field in amounts:
            amount = self._validate_amount(data.get(amount_field))
            if amount is not None:
                result[amount_field] = float(amount)
            elif data.get(amount_field):
                self.warnings.append(f"Invalid {amount_field}: {data.get(amount_field)}")
        
        return result
    
    def _validate_dates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate date fields"""
        result = {}
        
        expense_date = data.get("expense_date")
        if expense_date:
            parsed_date = self._parse_date(expense_date)
            if parsed_date:
                # Check if date is reasonable (not too far in past/future)
                now = datetime.now()
                years_diff = abs((now - parsed_date).days / 365.25)
                
                if years_diff > 10:
                    self.warnings.append(f"Expense date seems unusual: {parsed_date.date()}")
                
                result["expense_date"] = parsed_date
            else:
                self.warnings.append(f"Invalid expense date format: {expense_date}")
        
        return result
    
    def _validate_contact_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate contact information"""
        result = {}
        
        # Email validation
        email = self._clean_string(data.get("supplier_email"))
        if email:
            if self._validate_email(email):
                result["supplier_email"] = email.lower()
            else:
                self.warnings.append(f"Invalid email format: {email}")
        
        # Phone validation
        phone = self._clean_string(data.get("supplier_phone_number"))
        if phone:
            cleaned_phone = self._validate_phone(phone)
            if cleaned_phone:
                result["supplier_phone_number"] = cleaned_phone
            else:
                self.warnings.append(f"Invalid phone format: {phone}")
        
        # Invoice number
        invoice_number = self._clean_string(data.get("invoice_number"))
        if invoice_number:
            if len(invoice_number) > 100:
                self.warnings.append("Invoice number truncated")
                invoice_number = invoice_number[:100]
            result["invoice_number"] = invoice_number
        
        return result
    
    def _cross_validate_amounts(self, data: Dict[str, Any]):
        """Cross-validate financial calculations"""
        net = data.get("total_net")
        tax = data.get("total_tax")
        total = data.get("total_amount")
        
        if net is not None and tax is not None and total is not None:
            calculated_total = net + tax
            difference = abs(calculated_total - total)
            
            # Allow small rounding differences (0.02)
            if difference > 0.02:
                self.warnings.append(
                    f"Amount calculation mismatch: {net} + {tax} = {calculated_total}, "
                    f"but total_amount is {total} (difference: {difference})"
                )
        
        # Check for negative amounts
        for field in ["total_net", "total_tax", "total_amount"]:
            amount = data.get(field)
            if amount is not None and amount < 0:
                self.warnings.append(f"Negative amount detected in {field}: {amount}")
    
    def _clean_string(self, value) -> str:
        """Clean and normalize string values"""
        if not value:
            return None
        
        if isinstance(value, str):
            # Remove extra whitespace and normalize
            cleaned = ' '.join(value.strip().split())
            return cleaned if cleaned else None
        
        return str(value).strip() or None
    
    def _validate_amount(self, value) -> Decimal:
        """Validate and parse monetary amounts"""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                return None
        
        if isinstance(value, str):
            # Remove currency symbols and clean up
            cleaned = re.sub(r'[€$£¥₹,\s]', '', value.strip())
            try:
                return Decimal(cleaned)
            except (InvalidOperation, ValueError):
                return None
        
        return None
    
    def _parse_date(self, date_str) -> datetime:
        """Parse various date formats"""
        if not date_str:
            return None
        
        if isinstance(date_str, datetime):
            return date_str
        
        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
            "%Y/%m/%d", "%B %d, %Y", "%d %B %Y", "%d.%m.%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_phone(self, phone: str) -> str:
        """Validate and clean phone number"""
        # Remove all non-digit characters except + at the beginning
        cleaned = re.sub(r'[^\d+]', '', phone)
        if cleaned.startswith('+'):
            cleaned = '+' + re.sub(r'[^\d]', '', cleaned[1:])
        else:
            cleaned = re.sub(r'[^\d]', '', cleaned)
        
        # Check if it's a reasonable length (7-15 digits)
        digit_count = len(re.sub(r'[^\d]', '', cleaned))
        if 7 <= digit_count <= 15:
            return cleaned
        
        return None
    
    def _validate_vat_number(self, vat: str) -> str:
        """Basic VAT number validation"""
        # Remove spaces and convert to uppercase
        cleaned = re.sub(r'\s+', '', vat.upper())
        
        # Basic format check (2 letters + 8-12 digits for EU VAT)
        if re.match(r'^[A-Z]{2}[\dA-Z]{8,12}$', cleaned):
            return cleaned
        
        # Or just numbers (some countries)
        if re.match(r'^\d{8,12}$', cleaned):
            return cleaned
        
        return None
    
    def _validate_website(self, website: str) -> str:
        """Validate and clean website URL"""
        if not website.startswith(('http://', 'https://')):
            website = 'https://' + website
        
        # Basic URL pattern
        pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/?.*$'
        if re.match(pattern, website):
            return website
        
        return None