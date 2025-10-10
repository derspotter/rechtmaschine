"""
SQLAlchemy models for Rechtmaschine
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database import Base


class Document(Base):
    """Classified legal documents (uploaded PDFs)"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    explanation = Column(Text)
    file_path = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship to processed documents
    processed = relationship("ProcessedDocument", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "category": self.category,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "file_path": self.file_path,
            "timestamp": self.created_at.isoformat() if self.created_at else None
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
            "timestamp": self.created_at.isoformat() if self.created_at else None
        }


class ProcessedDocument(Base):
    """Processed documents with extracted/anonymized text (future feature)"""
    __tablename__ = "processed_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    extracted_text = Column(Text)
    is_anonymized = Column(Boolean, default=False)
    ocr_applied = Column(Boolean, default=False)
    anonymization_metadata = Column(JSONB)  # Store details about what was anonymized
    processing_status = Column(String(20), default="pending", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship to parent document
    document = relationship("Document", back_populates="processed")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "extracted_text": self.extracted_text,
            "is_anonymized": self.is_anonymized,
            "ocr_applied": self.ocr_applied,
            "anonymization_metadata": self.anonymization_metadata,
            "processing_status": self.processing_status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
