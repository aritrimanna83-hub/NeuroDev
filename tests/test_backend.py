from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.reasoning import ReasoningService


client = TestClient(app)


# ============================================================
# API tests
# ============================================================


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "NeuroDev"
    assert "version" in data


def test_health():
    with (
        patch("app.main.memory_service.redis.ping"),
        patch("app.main.psycopg.connect"),
    ):
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "NeuroDev"


def test_create_project():
    mock_connection = MagicMock()
    mock_cursor = MagicMock()

    mock_connection.__enter__.return_value = mock_connection
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    with patch(
        "app.main.psycopg.connect",
        return_value=mock_connection,
    ):
        response = client.post(
            "/projects",
            json={
                "name": "Test Project",
                "description": "NeuroDev test project",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "project_id" in data
    assert data["name"] == "Test Project"
    assert data["description"] == "NeuroDev test project"

    mock_cursor.execute.assert_called_once()


# ============================================================
# Reasoning tests
# ============================================================


def test_same_decision_is_consistent():
    reasoning = ReasoningService()

    result = reasoning.compare_decisions(
        new_decision="Use PostgreSQL",
        new_reason="Need relational storage",
        previous_decision="Use PostgreSQL",
        previous_reason="Project data is relational",
    )

    assert result["conflict_detected"] is False
    assert result["conflict_type"] == "consistent"


def test_direct_conflict_is_detected():
    reasoning = ReasoningService()

    result = reasoning.compare_decisions(
        new_decision="Use MongoDB",
        new_reason="Prefer document storage",
        previous_decision="Use PostgreSQL",
        previous_reason="Need relational storage",
    )

    assert result["conflict_detected"] is True
    assert result["conflict_type"] == "direct_contradiction"


def test_unrelated_decisions_do_not_conflict():
    reasoning = ReasoningService()

    result = reasoning.compare_decisions(
        new_decision="Use Redis for sessions",
        new_reason="Need fast temporary storage",
        previous_decision="Use PostgreSQL",
        previous_reason="Need relational project data",
    )

    assert result["conflict_detected"] is False


def test_confidence_for_developer_decision():
    reasoning = ReasoningService()

    confidence = reasoning.calculate_confidence(
        source="developer",
        similarity=1.0,
        has_reason=True,
    )

    assert confidence == 1.0


def test_high_confidence_conflict_requires_user():
    reasoning = ReasoningService()

    comparison = {
        "conflict_detected": True,
        "conflict_type": "direct_contradiction",
        "similarity": 0.8,
    }

    result = reasoning.evaluate_conflict(
        comparison=comparison,
        previous_confidence=0.95,
    )

    assert result["requires_user_decision"] is True
    assert result["severity"] == "high"


def test_low_confidence_conflict_does_not_require_user():
    reasoning = ReasoningService()

    comparison = {
        "conflict_detected": True,
        "conflict_type": "direct_contradiction",
        "similarity": 0.8,
    }

    result = reasoning.evaluate_conflict(
        comparison=comparison,
        previous_confidence=0.40,
    )

    assert result["requires_user_decision"] is False


# ============================================================
# Decision resolution tests
# ============================================================


def test_resolve_decision():
    reasoning = ReasoningService()

    result = reasoning.resolve_decision(
        selected_decision="Use PostgreSQL",
        reason="The project needs relational data.",
    )

    assert result["resolved"] is True
    assert result["resolved_by"] == "developer"
    assert result["final_decision"] == "Use PostgreSQL"