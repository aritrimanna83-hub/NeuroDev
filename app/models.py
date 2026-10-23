from typing import Any, Optional

from pydantic import BaseModel, Field # type: ignore


# ============================================================
# Chat / Developer Query
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Developer's question, idea, or technical decision."
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique conversation/session identifier."
    )
    project_id: str = Field(
        ...,
        min_length=1,
        description="Project associated with the conversation."
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
    project_id: str

    conflict_detected: bool = False
    requires_user_decision: bool = False

    confidence_score: float | None = None
    previous_decision: str | None = None
    new_decision: str | None = None


# ============================================================
# Developer Decision
# ============================================================

class DecisionRequest(BaseModel):
    session_id: str
    project_id: str

    decision: str = Field(
        ...,
        min_length=1,
        description="Developer's choice when resolving a decision conflict."
    )

    reason: str | None = Field(
        default=None,
        description="Optional explanation for the developer's choice."
    )


class DecisionResponse(BaseModel):
    success: bool
    message: str

    final_decision: str
    reason: str | None = None

    memory_saved: bool = False


# ============================================================
# Memory
# ============================================================

class MemoryRequest(BaseModel):
    session_id: str
    project_id: str

    content: str = Field(
        ...,
        min_length=1,
        description="Information or decision to remember."
    )

    memory_type: str = Field(
        default="general",
        description="Type of memory, e.g. decision, fact, insight, conversation."
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    success: bool
    message: str
    memory_id: str | None = None


# ============================================================
# Project
# ============================================================

class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    description: str | None = None


# ============================================================
# Health Check
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    dependencies: dict[str, str] = Field(default_factory=dict)