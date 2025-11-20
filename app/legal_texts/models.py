"""
Pydantic models for legal provision extraction and management.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LegalProvision(BaseModel):
    """Represents a specific legal provision (e.g., § 3 Abs. 1 AsylG)"""
    law: str = Field(..., description="Law abbreviation: AsylG, AufenthG, GG, etc.")
    paragraph: str = Field(..., description="Paragraph number: 3, 60, 16a, etc.")
    absatz: Optional[List[str]] = Field(None, description="Specific clauses: ['1', '2']")
    reasoning: str = Field(..., description="Why this provision is relevant to the case")

    def __str__(self) -> str:
        """Human-readable representation"""
        result = f"§ {self.paragraph} {self.law}"
        if self.absatz:
            result += f" Abs. {', '.join(self.absatz)}"
        return result


class ProvisionsExtractionResult(BaseModel):
    """Result of extracting both keywords and legal provisions from a query/document"""
    keywords: List[str] = Field(default_factory=list, description="asyl.net search keywords")
    provisions: List[LegalProvision] = Field(default_factory=list, description="Relevant legal provisions")
