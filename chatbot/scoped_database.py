"""Authenticated adapter for the user-scoped PostgreSQL database layer.

Legacy graph/tool functions call this module with their previous signatures.
The adapter injects only the verified identity from user_context; callers and
LLM-generated tool arguments cannot select another user's records.
"""

import database_postgres as _db
from user_context import get_current_user_id

_USER_SCOPED = {
    "validate_category_id", "get_categories", "get_categories_for_month",
    "create_category", "delete_category_for_month", "add_expense",
    "get_expenses", "get_expense_by_id", "update_expense",
    "update_expense_if_unchanged", "delete_expense",
    "initialise_category_budgets_for_month", "get_category_budgets",
    "get_category_budget", "update_category_budget", "get_monthly_summary",
    "get_category_summary", "get_daily_summary", "get_top_expenses",
    "search_expenses", "get_monthly_trend", "get_payment_method_summary",
    "get_dashboard_summary",
}

def __getattr__(name):
    """Resolve a database function and inject its trusted user_id when needed."""
    value = getattr(_db, name)
    if not callable(value) or name not in _USER_SCOPED:
        return value
    def scoped(*args, **kwargs):
        return value(get_current_user_id(), *args, **kwargs)
    scoped.__name__ = name
    scoped.__doc__ = getattr(value, "__doc__", None)
    return scoped
