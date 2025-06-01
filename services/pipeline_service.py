import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from services.minio_service import MinIOService
from services.data_validator import InvoiceDataValidator
from services.invoice_service import InvoiceService
from helpers.ocr_helper import extract_text_from_image
from Chain import extract_invoice_data

class PipelineService:
    def __init__(self):
        self.minio_service = MinIOService()
        self.validator = InvoiceDataValidator()
        self.db = SessionLocal()
        self.invoice_service = InvoiceService(self.db)
    
    def process_invoice_pipeline(self, invoice_id: int, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Complete pipeline for processing a single invoice:
        1. Upload raw image to MinIO
        2. Extract OCR text and upload to MinIO
        3. Extract structured data with LLM and upload to MinIO
        4. Validate and clean data
        5. Upload cleaned data to MinIO
        6. Save to database
        """
        
        pipeline_result = {
            "invoice_id": invoice_id,
            "stages": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Stage 1: Upload raw invoice
            print(f"Stage 1: Uploading raw invoice {invoice_id}")
            raw_object_name = self.minio_service.upload_raw_invoice(invoice_id, file_content, filename)
            pipeline_result["stages"]["raw_upload"] = {
                "status": "success" if raw_object_name else "failed",
                "object_name": raw_object_name
            }
            
            # Stage 2: OCR extraction
            print(f"Stage 2: Extracting OCR text for invoice {invoice_id}")
            
            # Save temp file for OCR processing
            temp_path = f"/tmp/invoice_{invoice_id}_{filename}"
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            try:
                ocr_text = extract_text_from_image(temp_path)
                ocr_object_name = self.minio_service.upload_ocr_output(invoice_id, ocr_text)
                pipeline_result["stages"]["ocr_extraction"] = {
                    "status": "success" if ocr_object_name else "failed",
                    "object_name": ocr_object_name,
                    "text_length": len(ocr_text) if ocr_text else 0
                }
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            if not ocr_text:
                raise Exception("OCR extraction failed")
            
            # Stage 3: LLM extraction
            print(f"Stage 3: Extracting structured data for invoice {invoice_id}")
            llm_result = extract_invoice_data(ocr_text=ocr_text)
            
            if isinstance(llm_result, str):
                try:
                    llm_data = json.loads(llm_result)
                except json.JSONDecodeError:
                    raise Exception("Invalid JSON from LLM")
            else:
                llm_data = llm_result
            
            llm_object_name = self.minio_service.upload_llm_output(invoice_id, llm_data)
            pipeline_result["stages"]["llm_extraction"] = {
                "status": "success" if llm_object_name else "failed",
                "object_name": llm_object_name
            }
            
            # Stage 4: Data validation and cleaning
            print(f"Stage 4: Validating and cleaning data for invoice {invoice_id}")
            cleaned_data, errors, warnings = self.validator.validate_invoice(llm_data)
            
            pipeline_result["errors"].extend(errors)
            pipeline_result["warnings"].extend(warnings)
            
            pipeline_result["stages"]["data_validation"] = {
                "status": "success" if not errors else "failed",
                "errors_count": len(errors),
                "warnings_count": len(warnings)
            }
            
            # Stage 5: Upload cleaned data
            print(f"Stage 5: Uploading cleaned data for invoice {invoice_id}")
            
            # Add validation results to cleaned data
            cleaned_data_with_validation = {
                **cleaned_data,
                "validation_results": {
                    "errors": errors,
                    "warnings": warnings,
                    "is_valid": len(errors) == 0
                }
            }
            
            cleaned_object_name = self.minio_service.upload_cleaned_invoice(invoice_id, cleaned_data_with_validation)
            pipeline_result["stages"]["cleaned_upload"] = {
                "status": "success" if cleaned_object_name else "failed",
                "object_name": cleaned_object_name
            }
            
            # Stage 6: Save to database
            print(f"Stage 6: Saving to database for invoice {invoice_id}")
            saved_invoice = self.invoice_service.create_invoice(cleaned_data, ocr_text)
            pipeline_result["stages"]["database_save"] = {
                "status": "success",
                "database_id": saved_invoice.id
            }
            
            pipeline_result["final_status"] = "success" if len(errors) == 0 else "success_with_warnings"
            pipeline_result["cleaned_data"] = cleaned_data
            
        except Exception as e:
            pipeline_result["final_status"] = "failed"
            pipeline_result["errors"].append(str(e))
            print(f"Pipeline failed for invoice {invoice_id}: {str(e)}")
        
        return pipeline_result
    
    def get_pipeline_status(self, invoice_id: int) -> Dict[str, Any]:
        """Get the processing status of an invoice from MinIO"""
        status = {
            "invoice_id": invoice_id,
            "stages": {}
        }
        
        # Check each stage
        buckets = self.minio_service.buckets
        
        # Check raw invoice
        raw_objects = self.minio_service.list_objects(buckets["raw_invoices"], f"invoice_{invoice_id}")
        status["stages"]["raw_upload"] = len(raw_objects) > 0
        
        # Check OCR output
        ocr_objects = self.minio_service.list_objects(buckets["ocr_output"], f"invoice_{invoice_id}")
        status["stages"]["ocr_extraction"] = len(ocr_objects) > 0
        
        # Check LLM output
        llm_objects = self.minio_service.list_objects(buckets["llm_output"], f"invoice_{invoice_id}")
        status["stages"]["llm_extraction"] = len(llm_objects) > 0
        
        # Check cleaned data
        cleaned_objects = self.minio_service.list_objects(buckets["cleaned_invoices"], f"invoice_{invoice_id}")
        status["stages"]["cleaned_data"] = len(cleaned_objects) > 0
        
        return status
    
    def get_cleaned_invoice_data(self, invoice_id: int) -> Dict[str, Any]:
        """Retrieve cleaned invoice data from MinIO"""
        object_name = f"invoice_{invoice_id}_cleaned.json"
        
        data = self.minio_service.get_object(
            self.minio_service.buckets["cleaned_invoices"], 
            object_name
        )
        
        if data:
            return json.loads(data.decode('utf-8'))
        
        return None
    
    def reprocess_invoice(self, invoice_id: int) -> Dict[str, Any]:
        """Reprocess an invoice through the validation and cleaning pipeline"""
        # Get LLM output from MinIO
        llm_object_name = f"invoice_{invoice_id}_llm.json"
        llm_data_bytes = self.minio_service.get_object(
            self.minio_service.buckets["llm_output"],
            llm_object_name
        )
        
        if not llm_data_bytes:
            return {"error": "No LLM output found for this invoice"}
        
        llm_data = json.loads(llm_data_bytes.decode('utf-8'))
        
        # Re-validate and clean
        cleaned_data, errors, warnings = self.validator.validate_invoice(llm_data)
        
        # Upload new cleaned data
        cleaned_data_with_validation = {
            **cleaned_data,
            "validation_results": {
                "errors": errors,
                "warnings": warnings,
                "is_valid": len(errors) == 0,
                "reprocessed_at": datetime.utcnow().isoformat()
            }
        }
        
        cleaned_object_name = self.minio_service.upload_cleaned_invoice(invoice_id, cleaned_data_with_validation)
        
        return {
            "status": "success",
            "errors": errors,
            "warnings": warnings,
            "cleaned_object_name": cleaned_object_name
        }
    
    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()