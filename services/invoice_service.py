from datetime import datetime
from sqlalchemy.orm import Session
from models.invoice import Invoice
from typing import Dict, Any, Optional, List
import re

class InvoiceService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_invoice(self, invoice_data: Dict[str, Any], ocr_text: str = None) -> Invoice:
        """Create a new invoice record from extracted data"""
        
        # Parse date if present
        expense_date = self._parse_date(invoice_data.get("expense_date"))
        
        invoice = Invoice(
            supplier_name=self._clean_string(invoice_data.get("supplier_name")),
            supplier_address=self._clean_string(invoice_data.get("supplier_address")),
            supplier_email=self._clean_string(invoice_data.get("supplier_email")),
            supplier_phone_number=self._clean_string(invoice_data.get("supplier_phone_number")),
            supplier_vat_number=self._clean_string(invoice_data.get("supplier_vat_number")),
            supplier_website=self._clean_string(invoice_data.get("supplier_website")),
            expense_date=expense_date,
            invoice_number=self._clean_string(invoice_data.get("invoice_number")),
            currency=self._clean_string(invoice_data.get("currency")),
            total_net=self._parse_numeric(invoice_data.get("total_net")),
            total_tax=self._parse_numeric(invoice_data.get("total_tax")),
            total_amount=self._parse_numeric(invoice_data.get("total_amount")),
            original_ocr_text=ocr_text
        )
        
        self.db.add(invoice)
        self.db.commit()
        self.db.refresh(invoice)
        return invoice
    
    def get_invoice(self, invoice_id: int) -> Optional[Invoice]:
        """Get a single invoice by ID"""
        return self.db.query(Invoice).filter(Invoice.id == invoice_id).first()
    
    def get_invoices(self, skip: int = 0, limit: int = 100) -> List[Invoice]:
        """Get list of invoices with pagination"""
        return self.db.query(Invoice).offset(skip).limit(limit).all()
    
    def search_invoices(self, query: str) -> List[Invoice]:
        """Search invoices by supplier name or invoice number"""
        return self.db.query(Invoice).filter(
            (Invoice.supplier_name.contains(query)) |
            (Invoice.invoice_number.contains(query))
        ).all()
    
    def update_invoice(self, invoice_id: int, invoice_data: Dict[str, Any]) -> Optional[Invoice]:
        """Update an existing invoice"""
        invoice = self.get_invoice(invoice_id)
        if not invoice:
            return None
        
        # Update fields
        for key, value in invoice_data.items():
            if hasattr(invoice, key):
                if key == "expense_date":
                    value = self._parse_date(value)
                elif key in ["total_net", "total_tax", "total_amount"]:
                    value = self._parse_numeric(value)
                elif isinstance(value, str):
                    value = self._clean_string(value)
                
                setattr(invoice, key, value)
        
        invoice.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(invoice)
        return invoice
    
    def delete_invoice(self, invoice_id: int) -> bool:
        """Delete an invoice"""
        invoice = self.get_invoice(invoice_id)
        if not invoice:
            return False
        
        self.db.delete(invoice)
        self.db.commit()
        return True
    
    def _parse_date(self, date_str) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_str:
            return None
        
        if isinstance(date_str, datetime):
            return date_str
        
        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%B %d, %Y",
            "%d %B %Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def _parse_numeric(self, value) -> Optional[float]:
        """Parse numeric values from strings, handling currency symbols"""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove currency symbols, commas, and whitespace
            cleaned = re.sub(r'[€$£¥₹,\s]', '', value.strip())
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def _clean_string(self, value) -> Optional[str]:
        """Clean and normalize string values"""
        if not value:
            return None
        
        if isinstance(value, str):
            return value.strip() or None
        
        return str(value).strip() or None