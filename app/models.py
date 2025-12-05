"""
SQLAlchemy models for Rechtmaschine
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Text, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from database import Base


class Document(Base):
    """Classified legal documents (uploaded PDFs) with OCR and anonymization data"""
    __tablename__ = "documents"

    # Core document fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    explanation = Column(Text)
    file_path = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # OCR and anonymization fields
    extracted_text_path = Column(String(512))
    is_anonymized = Column(Boolean, default=False)
    ocr_applied = Column(Boolean, default=False)
    needs_ocr = Column(Boolean, default=False)
    anonymization_metadata = Column(JSONB)
    processing_status = Column(String(20), default='pending')
    gemini_file_uri = Column(String(255))
    owner_id = Column(UUID(as_uuid=True), index=True)  # ForeignKey("users.id") added later to avoid circular deps if needed, but usually fine.

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "category": self.category,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "file_path": self.file_path,
            "extracted_text_path": self.extracted_text_path,
            "timestamp": self.created_at.isoformat() if self.created_at else None,
            "anonymized": self.is_anonymized or False,
            "needs_ocr": self.needs_ocr or False,
            "ocr_applied": self.ocr_applied or False,
            "gemini_file_uri": self.gemini_file_uri,
            "owner_id": str(self.owner_id) if self.owner_id else None
        }


class ResearchSource(Base):
    """Legal research sources (court decisions, articles, etc.)"""
    __tablename__ = "research_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(512), nullable=False)
    url = Column(Text, nullable=False)
    description = Column(Text)
    document_type = Column(String(50), index=True)
    pdf_url = Column(Text)
    download_path = Column(String(512))
    download_status = Column(String(20), default="pending", index=True)
    research_query = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    gemini_file_uri = Column(String(255))
    owner_id = Column(UUID(as_uuid=True), index=True)

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "document_type": self.document_type,
            "pdf_url": self.pdf_url,
            "download_path": self.download_path,
            "download_status": self.download_status,
            "research_query": self.research_query,
            "timestamp": self.created_at.isoformat() if self.created_at else None,
            "owner_id": str(self.owner_id) if self.owner_id else None
        }


class User(Base):
    """User account model"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "is_active": self.is_active
        }
