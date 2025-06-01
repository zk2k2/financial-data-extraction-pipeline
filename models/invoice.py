from sqlalchemy import Column, Integer, String, DateTime, Numeric, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Invoice(Base):
    __tablename__ = "invoices"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Supplier information
    supplier_name = Column(String(255), nullable=True, index=True)
    supplier_address = Column(Text, nullable=True)
    supplier_email = Column(String(255), nullable=True)
    supplier_phone_number = Column(String(50), nullable=True)
    supplier_vat_number = Column(String(50), nullable=True)
    supplier_website = Column(String(255), nullable=True)
    
    # Invoice details
    expense_date = Column(DateTime, nullable=True, index=True)
    invoice_number = Column(String(100), nullable=True, index=True)
    currency = Column(String(10), nullable=True)
    
    # Financial amounts - using Numeric for precision
    total_net = Column(Numeric(12, 2), nullable=True)
    total_tax = Column(Numeric(12, 2), nullable=True)
    total_amount = Column(Numeric(12, 2), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Store original OCR text for reference
    original_ocr_text = Column(Text, nullable=True)
    
    def to_dict(self):
        """Convert model instance to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "supplier_name": self.supplier_name,
            "supplier_address": self.supplier_address,
            "supplier_email": self.supplier_email,
            "supplier_phone_number": self.supplier_phone_number,
            "supplier_vat_number": self.supplier_vat_number,
            "supplier_website": self.supplier_website,
            "expense_date": self.expense_date.isoformat() if self.expense_date else None,
            "invoice_number": self.invoice_number,
            "currency": self.currency,
            "total_net": float(self.total_net) if self.total_net else None,
            "total_tax": float(self.total_tax) if self.total_tax else None,
            "total_amount": float(self.total_amount) if self.total_amount else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f"<Invoice(id={self.id}, supplier='{self.supplier_name}', invoice_number='{self.invoice_number}')>"