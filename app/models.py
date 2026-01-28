"""
SQLAlchemy models for Rechtmaschine
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Text, DateTime, Boolean, ForeignKey, Date
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


class GeneratedDraft(Base):
    """Persisted generated legal drafts"""
    __tablename__ = "generated_drafts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    primary_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    document_type = Column(String(100), nullable=False)
    user_prompt = Column(Text)
    generated_text = Column(Text, nullable=False)
    model_used = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata_ = Column("metadata", JSONB, default={}) # Using metadata_ to avoid conflict with Base.metadata

    def to_dict(self):
        return {
            "id": str(self.id),
            "document_type": self.document_type,
            "user_prompt": self.user_prompt,
            "generated_text": self.generated_text,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "primary_document_id": str(self.primary_document_id) if self.primary_document_id else None,
            "metadata": self.metadata_
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


class RechtsprechungEntry(Base):
    """Curated Aktuelle Rechtsprechung entries derived from Rechtsprechung documents."""
    __tablename__ = "rechtsprechung_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    country = Column(String(100), nullable=False, index=True)
    tags = Column(JSONB, default=list)
    court = Column(String(255))
    court_level = Column(String(50))
    decision_date = Column(Date)
    aktenzeichen = Column(String(100))
    outcome = Column(String(50))
    key_facts = Column(JSONB, default=list)
    key_holdings = Column(JSONB, default=list)
    argument_patterns = Column(JSONB, default=list)
    citations = Column(JSONB, default=list)
    summary = Column(Text)
    extracted_at = Column(DateTime)
    model = Column(String(50))
    confidence = Column(Float)
    warnings = Column(JSONB, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self):
        return {
            "id": str(self.id),
            "document_id": str(self.document_id) if self.document_id else None,
            "country": self.country,
            "tags": self.tags or [],
            "court": self.court,
            "court_level": self.court_level,
            "decision_date": self.decision_date.isoformat() if self.decision_date else None,
            "aktenzeichen": self.aktenzeichen,
            "outcome": self.outcome,
            "key_facts": self.key_facts or [],
            "key_holdings": self.key_holdings or [],
            "argument_patterns": self.argument_patterns or [],
            "citations": self.citations or [],
            "summary": self.summary,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
            "model": self.model,
            "confidence": self.confidence,
            "warnings": self.warnings or [],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
