"""Focused tests for FinSight account deletion.

These tests mock Supabase Auth and startup services; they never contact a real
Supabase project or delete a real account.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def app_module(monkeypatch):
    # Ensure the repository root is importable regardless of pytest invocation.
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Stub optional startup modules absent from the test checkout and avoid
    # real DB/checkpointer initialization.
    import types
    database_stub = types.ModuleType("database_postgres")
    database_stub.initialise_database = lambda: None
    monkeypatch.setitem(sys.modules, "database_postgres", database_stub)

    # Stub the user-scoped database adapter as app imports its functions at startup.
    scoped_stub = types.ModuleType("chatbot.scoped_database")
    for name in (
        "get_categories_for_month", "create_category", "delete_category_for_month",
        "get_category_budgets", "update_category_budget", "add_expense",
        "get_expenses", "search_expenses", "get_expense_by_id", "update_expense",
        "delete_expense", "get_monthly_summary", "get_dashboard_summary",
        "get_category_summary", "get_daily_summary", "get_top_expenses",
        "get_payment_method_summary", "get_monthly_trend",
    ):
        setattr(scoped_stub, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "chatbot.scoped_database", scoped_stub)

    # Chatbot graph startup is unrelated to account-deletion route tests.
    graph_stub = types.ModuleType("chatbot.graph")
    graph_stub.build_chatbot_graph = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "chatbot.graph", graph_stub)

    memory_stub = types.ModuleType("chatbot.memory")
    memory_stub.initialise_checkpointer = lambda: None
    monkeypatch.setitem(sys.modules, "chatbot.memory", memory_stub)

    if "app" in sys.modules:
        module = importlib.reload(sys.modules["app"])
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
