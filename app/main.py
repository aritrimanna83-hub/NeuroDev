from __future__ import annotations

from fastapi import FastAPI, HTTPException
import psycopg

from app.config import settings
from app.memory import memory_service
from app.models import (
    ChatRequest,
    ChatResponse,
    DecisionRequest,
    DecisionResponse,
    HealthResponse,
    MemoryRequest,
    MemoryResponse,
    ProjectCreateRequest,
    ProjectResponse,
)
from app.workflow import neurodev_workflow


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="NeuroDev - AI-powered second brain for developers.",
)


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    dependencies = {}

    # Redis
    try:
        memory_service.redis.ping()
        dependencies["redis"] = "connected"
    except Exception:
        dependencies["redis"] = "disconnected"

    # PostgreSQL
    try:
        with psycopg.connect(settings.POSTGRES_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
        dependencies["postgresql"] = "connected"
    except Exception:
        dependencies["postgresql"] = "disconnected"

    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        dependencies=dependencies,
    )


# -------------------------------------------------------------------
# Chat
# -------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Start a new NeuroDev conversation.

    The session_id is also used as the LangGraph thread ID so that
    a paused decision conflict can later be resumed using /decision.
    """

    try:
        # Save the user's message in short-term memory.
        memory_service.save_session_message(
            session_id=request.session_id,
            role="user",
            content=request.question,
        )

        initial_state = {
            "question": request.question,
            "session_id": request.session_id,
            "project_id": request.project_id,
        }

        config = {
            "configurable": {
                "thread_id": request.session_id,
            }
        }

        result = neurodev_workflow.invoke(
            initial_state,
            config=config,
        )

        response = result.get(
            "final_response",
            "I could not generate a response.",
        )

        # Save assistant response in short-term memory.
        memory_service.save_session_message(
            session_id=request.session_id,
            role="assistant",
            content=response,
        )

        return ChatResponse(
            response=response,
            session_id=request.session_id,
            project_id=request.project_id,
            conflict_detected=result.get(
                "conflict_detected",
                False,
            ),
            requires_user_decision=result.get(
                "requires_user_decision",
                False,
            ),
            confidence_score=result.get(
                "confidence_score"
            ),
            previous_decision=result.get(
                "previous_decision"
            ),
            new_decision=result.get(
                "new_decision"
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(exc)}",
        ) from exc


# -------------------------------------------------------------------
# Decision resolution
# -------------------------------------------------------------------

@app.post("/decision", response_model=DecisionResponse)
def resolve_decision(
    request: DecisionRequest,
) -> DecisionResponse:
    """
    Resume a paused LangGraph workflow after the developer
    resolves a decision conflict.
    """

    config = {
        "configurable": {
            "thread_id": request.session_id,
        }
    }

    try:
        # Resume the interrupted human node.
        result = neurodev_workflow.invoke(
            {
                "decision": request.decision,
                "reason": request.reason,
            },
            config={
                **config,
                "resume": {
                    "decision": request.decision,
                    "reason": request.reason,
                },
            },
        )

        final_decision = result.get(
            "final_decision",
            request.decision,
        )

        final_response = result.get(
            "final_response",
            f"Decision resolved: {final_decision}",
        )

        # Store the final assistant response in short-term memory.
        memory_service.save_session_message(
            session_id=request.session_id,
            role="assistant",
            content=final_response,
        )

        return DecisionResponse(
            success=True,
            message=final_response,
            final_decision=final_decision,
            reason=request.reason,
            memory_saved=result.get(
                "memory_saved",
                False,
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Decision resolution failed: {str(exc)}",
        ) from exc


# -------------------------------------------------------------------
# Memory
# -------------------------------------------------------------------

@app.post("/memory", response_model=MemoryResponse)
def save_memory(request: MemoryRequest) -> MemoryResponse:
    """
    Manually save a piece of long-term project memory.
    """

    try:
        memory_id = memory_service.save_memory(
            project_id=request.project_id,
            content=request.content,
            memory_type=request.memory_type,
            metadata={
                **request.metadata,
                "session_id": request.session_id,
                "source": "developer",
            },
        )

        return MemoryResponse(
            success=True,
            message="Memory saved successfully.",
            memory_id=memory_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Memory save failed: {str(exc)}",
        ) from exc


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

@app.post("/projects", response_model=ProjectResponse)
def create_project(
    request: ProjectCreateRequest,
) -> ProjectResponse:
    import uuid

    project_id = str(uuid.uuid4())

    try:
        with psycopg.connect(settings.POSTGRES_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO projects (
                        id,
                        name,
                        description
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        project_id,
                        request.name,
                        request.description,
                    ),
                )
            conn.commit()

        return ProjectResponse(
            project_id=project_id,
            name=request.name,
            description=request.description,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Project creation failed: {str(exc)}",
        ) from exc


# -------------------------------------------------------------------
# Root
# -------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "message": "NeuroDev backend is running.",
    }