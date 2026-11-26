"""Focused tests for FinSight account deletion.

These tests mock Supabase Auth and startup services; they never contact a real
Supabase project or delete a real account.
"""
import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def app_module(monkeypatch):
    # Prevent application startup from opening a real database/checkpointer.
    import database_postgres
    import chatbot.memory

    monkeypatch.setattr(database_postgres, "initialise_database", lambda: None)
    monkeypatch.setattr(chatbot.memory, "initialise_checkpointer", lambda: None)

    if "app" in sys.modules:
        module = sys.modules["app"]
        monkeypatch.setattr(module, "initialise_database", lambda: None, raising=False)
        monkeypatch.setattr(module, "initialise_checkpointer", lambda: None, raising=False)
        module = importlib.reload(module)
    else:
        module = importlib.import_module("app")
    return module


@pytest.fixture
def client(app_module, monkeypatch):
    import auth

    monkeypatch.setattr(auth, "get_supabase_user", lambda token: {
        "id": "11111111-1111-4111-8111-111111111111",
        "email": "test@example.com",
        "name": "Test User",
    })
    monkeypatch.setattr(auth, "_approval_status", lambda user_id: "approved")
    return app_module.app.test_client()


def test_delete_account_success(client, app_module, monkeypatch):
    import auth
    monkeypatch.setenv("SUPABASE_URL", "https://isolated-project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
    monkeypatch.setenv("FINANCE_ADMIN_USER_IDS", "")

    calls = []
    monkeypatch.setattr(app_module.requests, "delete", lambda url, **kwargs: (
        calls.append((url, kwargs)) or SimpleNamespace(status_code=204)
    ))

    response = client.post("/api/account/delete", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"
    assert calls
    assert calls[0][0].endswith("/auth/v1/admin/users/11111111-1111-4111-8111-111111111111")
    assert calls[0][1]["headers"]["apikey"] == "test-service-key"


def test_delete_account_unauthorized(client, app_module, monkeypatch):
    response = client.post("/api/account/delete")
    assert response.status_code == 401


def test_delete_account_supabase_failure(client, app_module, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://isolated-project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
    monkeypatch.setenv("FINANCE_ADMIN_USER_IDS", "")
    monkeypatch.setattr(app_module.requests, "delete", lambda *args, **kwargs: SimpleNamespace(status_code=500))

    response = client.post("/api/account/delete", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 502
    assert response.get_json()["status"] == "error"


def test_delete_account_protects_configured_admin(client, app_module, monkeypatch):
    monkeypatch.setenv("FINANCE_ADMIN_USER_IDS", "11111111-1111-4111-8111-111111111111")
    calls = []
    monkeypatch.setattr(app_module.requests, "delete", lambda *args, **kwargs: calls.append(args))

    response = client.post("/api/account/delete", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 403
    assert response.get_json()["status"] == "error"
    assert calls == []
