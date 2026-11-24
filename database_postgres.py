"""FinSight PostgreSQL database layer.

This is a fresh PostgreSQL implementation of the existing SQLite database
API. Every function that reads or changes user data requires ``user_id`` as
its first argument. Pass only the UUID obtained from verified Supabase Auth
(e.g. ``flask.g.user_id``), never a value supplied by the browser or chatbot.

Database schema expected:
    public.categories
    public.expenses
    public.category_budgets
    public.category_periods

The schema is created by the first Supabase migration. This module does not
import old SQLite records or create/alter tables automatically.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from postgres_connection import health_connection, user_connection


# ---------------------------------------------------------------------------
# Shared validation and conversion helpers
# ---------------------------------------------------------------------------

def _require_user_id(user_id: str) -> str:
    """Reject missing identities before opening a database connection."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A verified authenticated user_id is required.")
    return user_id.strip()


def _parse_expense_date(expense_date: str) -> date:
    """Validate the public DD-MM-YYYY format and return a Python date."""
    if not isinstance(expense_date, str):
        raise ValueError("Date must be in DD-MM-YYYY format.")
    try:
        return datetime.strptime(expense_date, "%d-%m-%Y").date()
    except ValueError as exc:
        raise ValueError("Date must be in DD-MM-YYYY format.") from exc


def validate_expense_date(expense_date: str) -> str:
    """Validate an expense date while preserving the caller's string."""
    _parse_expense_date(expense_date)
    return expense_date


def validate_actual_expense_date(expense_date: str) -> str:
    """Validate DD-MM-YYYY and reject dates later than today."""
    parsed = _parse_expense_date(expense_date)
    if parsed > date.today():
        raise ValueError("Expense date cannot be in the future.")
    return expense_date


def _parse_budget_month(budget_month: str) -> date:
    """Convert a valid MM-YYYY value to the first day of that month."""
    if not isinstance(budget_month, str):
        raise ValueError("Budget month must be in MM-YYYY format.")
    try:
        parsed = datetime.strptime(budget_month, "%m-%Y").date()
    except ValueError as exc:
        raise ValueError(
            "Invalid budget month. Please provide a month between 01 and 12 "
            "in MM-YYYY format."
        ) from exc
    return parsed.replace(day=1)


def validate_budget_month(budget_month: str) -> str:
    """Validate the public MM-YYYY budget-month format."""
    _parse_budget_month(budget_month)
    return budget_month


def _validate_positive_id(value: Any, label: str = "Category ID") -> int:
    """Convert a positive integer-like identifier without accepting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a valid integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return parsed


def _validate_amount(value: Any, label: str, *, allow_zero: bool = False) -> Decimal:
    """Parse a finite money amount and avoid binary floating-point storage."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a valid number.")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid number.") from exc
    if not amount.is_finite():
        raise ValueError(f"{label} must be a valid number.")
    if amount < 0 or (amount == 0 and not allow_zero):
        message = f"{label} cannot be negative." if allow_zero else f"{label} must be greater than zero."
        raise ValueError(message)
    return amount.quantize(Decimal("0.01"))


def _display_expense_date(value: date | datetime | str | None) -> str | None:
    """Return PostgreSQL dates in the DD-MM-YYYY format used by the UI."""
    if value is None:
        return None
    if isinstance(value, str):
        value = date.fromisoformat(value[:10])
    return value.strftime("%d-%m-%Y")


def _normalise_expense_row(row: dict | None) -> dict | None:
    """Convert database date and Decimal values into JSON-friendly values."""
    if row is None:
        return None
    result = dict(row)
    if "expense_date" in result:
        result["expense_date"] = _display_expense_date(result["expense_date"])
    for key in ("amount", "budget_amount", "total_spending", "total_expenses",
                "average_expense", "budget_amount", "remaining_budget",
                "utilisation_percentage"):
        if key in result and isinstance(result[key], Decimal):
            result[key] = float(result[key])
    return result


def _validate_date_range(start_date: str | None, end_date: str | None) -> tuple[date | None, date | None]:
    """Validate optional date bounds and ensure start is not after end."""
    start = _parse_expense_date(start_date) if start_date else None
    end = _parse_expense_date(end_date) if end_date else None
    if start and end and start > end:
        raise ValueError("Start date cannot be after end date.")
    return start, end


def _validate_category_owned(conn, user_id: str, category_id: Any) -> int:
    """Confirm the category belongs to this user, not merely that it exists."""
    category_id = _validate_positive_id(category_id)
    row = conn.execute(
        """
        SELECT category_id
        FROM public.categories
        WHERE user_id = %s AND category_id = %s
        """,
        (user_id, category_id),
    ).fetchone()
    if row is None:
        raise ValueError("The selected category does not exist.")
    return category_id


def _ensure_month_category_rows(conn, user_id: str, category_id: int, month_start: date) -> None:
    """Activate a category and create its initial zero budget for a month."""
    conn.execute(
        """
        INSERT INTO public.category_periods (category_id, user_id, budget_month)
        VALUES (%s, %s, %s)
        ON CONFLICT (category_id, budget_month) DO NOTHING
        """,
        (category_id, user_id, month_start),
    )
    conn.execute(
        """
        INSERT INTO public.category_budgets
            (category_id, user_id, budget_month, budget_amount)
        VALUES (%s, %s, %s, 0)
        ON CONFLICT (category_id, budget_month) DO NOTHING
        """,
        (category_id, user_id, month_start),
    )


def _expense_select() -> str:
    """Shared expense projection used by list/detail/search operations."""
    return """
        SELECT e.expense_id, e.expense_name, e.amount, e.category_id,
               c.category_name, e.expense_date, e.payment_method, e.notes,
               e.created_at
        FROM public.expenses AS e
        JOIN public.categories AS c
          ON c.category_id = e.category_id AND c.user_id = e.user_id
    """


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def initialise_database() -> None:
    """Verify the Supabase schema exists without creating or changing tables.

    The original Flask application may call this function during startup.
    The PostgreSQL schema is managed explicitly in Supabase SQL Editor, so
    startup performs only a metadata check and never touches user records.
    """
    expected_tables = (
        "profiles",
        "categories",
        "expenses",
        "category_budgets",
        "category_periods",
    )
    with health_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                to_regclass('public.profiles') AS profiles,
                to_regclass('public.categories') AS categories,
                to_regclass('public.expenses') AS expenses,
                to_regclass('public.category_budgets') AS category_budgets,
                to_regclass('public.category_periods') AS category_periods
            """
        ).fetchone()

    missing = [name for name in expected_tables if rows[name] is None]
    if missing:
        raise RuntimeError(
            "Supabase schema is incomplete. Missing tables: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

def validate_category_id(user_id: str, category_id: Any) -> int:
    """Validate that a category exists and belongs to the authenticated user."""
    user_id = _require_user_id(user_id)
    with user_connection(user_id) as conn:
        return _validate_category_owned(conn, user_id, category_id)


def get_categories(user_id: str) -> list[dict]:
    """Retrieve the authenticated user's master category list."""
    user_id = _require_user_id(user_id)
    with user_connection(user_id) as conn:
        rows = conn.execute(
            """
            SELECT category_id, category_name
            FROM public.categories
            WHERE user_id = %s
            ORDER BY category_name
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_categories_for_month(user_id: str, budget_month: str) -> list[dict]:
    """Return categories active in a month, plus the default Food category."""
    user_id = _require_user_id(user_id)
    month_start = _parse_budget_month(budget_month)
    with user_connection(user_id) as conn:
        rows = conn.execute(
            """
            SELECT c.category_id, c.category_name
            FROM public.categories AS c
            WHERE c.user_id = %s
              AND (
                    c.category_name = 'Food'
                    OR EXISTS (
                        SELECT 1
                        FROM public.category_periods AS cp
                        WHERE cp.category_id = c.category_id
                          AND cp.user_id = c.user_id
                          AND cp.budget_month = %s
                    )
              )
            ORDER BY c.category_name
            """,
            (user_id, month_start),
        ).fetchall()
        return [dict(row) for row in rows]


def create_category(user_id: str, category_name: str, budget_month: str | None = None) -> int:
    """Create or reuse a user-owned category and optionally activate a month."""
    user_id = _require_user_id(user_id)
    if not isinstance(category_name, str) or not category_name.strip():
        raise ValueError("Category name must be non-empty text.")
    category_name = category_name.strip()
    if len(category_name) > 40:
        raise ValueError("Category name cannot exceed 40 characters.")
    month_start = _parse_budget_month(budget_month) if budget_month else None

    with user_connection(user_id) as conn:
        row = conn.execute(
            """
            SELECT category_id
            FROM public.categories
            WHERE user_id = %s AND LOWER(category_name) = LOWER(%s)
            """,
            (user_id, category_name),
        ).fetchone()
        if row:
            category_id = row["category_id"]
        else:
            row = conn.execute(
                """
                INSERT INTO public.categories (user_id, category_name)
                VALUES (%s, %s)
                RETURNING category_id
                """,
                (user_id, category_name),
            ).fetchone()
            category_id = row["category_id"]

        if month_start:
            _ensure_month_category_rows(conn, user_id, category_id, month_start)
        return category_id


def delete_category_for_month(user_id: str, category_id: Any, budget_month: str) -> bool:
    """Remove a category's availability and budget for one month only."""
    user_id = _require_user_id(user_id)
    month_start = _parse_budget_month(budget_month)
    with user_connection(user_id) as conn:
        category_id = _validate_category_owned(conn, user_id, category_id)
        category = conn.execute(
            """
            SELECT category_name FROM public.categories
            WHERE user_id = %s AND category_id = %s
            """,
            (user_id, category_id),
        ).fetchone()
        if category and category["category_name"] == "Food":
            raise ValueError("The default Food category cannot be removed.")

        count = conn.execute(
            """
            SELECT COUNT(*) AS expense_count
            FROM public.expenses
            WHERE user_id = %s
              AND category_id = %s
              AND expense_date >= %s
              AND expense_date < (%s + INTERVAL '1 month')
            """,
            (user_id, category_id, month_start, month_start),
        ).fetchone()["expense_count"]
        if count:
            raise ValueError(
                "This category has transactions in the selected month. "
                "Delete those transactions before removing the category."
            )

        conn.execute(
            """
            DELETE FROM public.category_periods
            WHERE user_id = %s AND category_id = %s AND budget_month = %s
            """,
            (user_id, category_id, month_start),
        )
        conn.execute(
            """
            DELETE FROM public.category_budgets
            WHERE user_id = %s AND category_id = %s AND budget_month = %s
            """,
            (user_id, category_id, month_start),
        )
        return True


# ---------------------------------------------------------------------------
# Expenses
# ---------------------------------------------------------------------------

def add_expense(
    user_id: str,
    expense_name: str,
    amount: Any,
    category_id: Any,
    expense_date: str,
    payment_method: str,
    notes: str | None = None,
) -> int:
    """Insert an expense and activate its category for that month."""
    user_id = _require_user_id(user_id)
    if not isinstance(expense_name, str) or not expense_name.strip():
        raise ValueError("Expense name must be non-empty text.")
    amount_value = _validate_amount(amount, "Amount")
    if not isinstance(payment_method, str) or not payment_method.strip():
        raise ValueError("Payment method must be non-empty text.")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("Notes must be text.")
    parsed_date = _parse_expense_date(expense_date)
    if parsed_date > date.today():
        raise ValueError("Expense date cannot be in the future.")
    month_start = parsed_date.replace(day=1)

    with user_connection(user_id) as conn:
        category_id = _validate_category_owned(conn, user_id, category_id)
        _ensure_month_category_rows(conn, user_id, category_id, month_start)
        row = conn.execute(
            """
            INSERT INTO public.expenses
                (user_id, expense_name, amount, category_id, expense_date,
                 payment_method, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING expense_id
            """,
            (
                user_id, expense_name.strip(), amount_value, category_id,
                parsed_date, payment_method.strip(), notes,
            ),
        ).fetchone()
        return row["expense_id"]


def get_expenses(user_id: str) -> list[dict]:
    """Retrieve all expenses belonging to the authenticated user."""
    user_id = _require_user_id(user_id)
    with user_connection(user_id) as conn:
        rows = conn.execute(
            _expense_select()
            + """
            WHERE e.user_id = %s
            ORDER BY e.expense_date DESC, e.expense_id DESC
            """,
            (user_id,),
        ).fetchall()
        return [_normalise_expense_row(row) for row in rows]


def get_expense_by_id(user_id: str, expense_id: Any) -> dict | None:
    """Retrieve one expense, constrained by both user and expense ID."""
    user_id = _require_user_id(user_id)
    expense_id = _validate_positive_id(expense_id, "Expense ID")
    with user_connection(user_id) as conn:
        row = conn.execute(
            _expense_select()
            + """
            WHERE e.user_id = %s AND e.expense_id = %s
            """,
            (user_id, expense_id),
        ).fetchone()
        return _normalise_expense_row(row)


def update_expense(
    user_id: str,
    expense_id: Any,
    expense_name: str,
    amount: Any,
    category_id: Any,
    expense_date: str,
    payment_method: str,
    notes: str | None = None,
) -> bool:
    """Update an expense only when it belongs to the authenticated user."""
    user_id = _require_user_id(user_id)
    expense_id = _validate_positive_id(expense_id, "Expense ID")
    if not isinstance(expense_name, str) or not expense_name.strip():
        raise ValueError("Expense name must be non-empty text.")
    amount_value = _validate_amount(amount, "Amount")
    if not isinstance(payment_method, str) or not payment_method.strip():
        raise ValueError("Payment method must be non-empty text.")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("Notes must be text.")
    parsed_date = _parse_expense_date(expense_date)
    if parsed_date > date.today():
        raise ValueError("Expense date cannot be in the future.")
    month_start = parsed_date.replace(day=1)

    with user_connection(user_id) as conn:
        category_id = _validate_category_owned(conn, user_id, category_id)
        _ensure_month_category_rows(conn, user_id, category_id, month_start)
        result = conn.execute(
            """
            UPDATE public.expenses
            SET expense_name = %s, amount = %s, category_id = %s,
                expense_date = %s, payment_method = %s, notes = %s
            WHERE user_id = %s AND expense_id = %s
            """,
            (
                expense_name.strip(), amount_value, category_id, parsed_date,
                payment_method.strip(), notes, user_id, expense_id,
            ),
        )
        return result.rowcount > 0


def update_expense_if_unchanged(
    user_id: str,
    expense_id: Any,
    expected_original: dict,
    proposed: dict,
) -> str:
    """Optimistically update an expense only if its reviewed snapshot is current.

    Returns ``updated``, ``not_found`` or ``changed``.
    """
    user_id = _require_user_id(user_id)
    expense_id = _validate_positive_id(expense_id, "Expense ID")
    required = {
        "expense_name", "amount", "category_id", "expense_date",
        "payment_method", "notes",
    }
    if not isinstance(expected_original, dict) or not isinstance(proposed, dict):
        raise ValueError("Original and proposed expense details must be dictionaries.")
    if not required.issubset(expected_original) or not required.issubset(proposed):
        raise ValueError("Original or proposed expense details are incomplete.")

    # Validate and normalise the proposed values before acquiring the row lock.
    name = proposed["expense_name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Expense name must be non-empty text.")
    amount_value = _validate_amount(proposed["amount"], "Amount")
    category_id = _validate_positive_id(proposed["category_id"])
    parsed_date = _parse_expense_date(proposed["expense_date"])
    if parsed_date > date.today():
        raise ValueError("Expense date cannot be in the future.")
    payment = proposed["payment_method"]
    notes = proposed["notes"]
    if not isinstance(payment, str) or not payment.strip():
        raise ValueError("Payment method must be non-empty text.")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("Notes must be text.")

    original_date = _parse_expense_date(expected_original["expense_date"])
    try:
        original_amount = Decimal(str(expected_original["amount"]))
        original_category = int(expected_original["category_id"])
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Original expense values are invalid.") from exc

    with user_connection(user_id) as conn:
        _validate_category_owned(conn, user_id, category_id)
        current = conn.execute(
            """
            SELECT expense_name, amount, category_id, expense_date,
                   payment_method, notes
            FROM public.expenses
            WHERE user_id = %s AND expense_id = %s
            FOR UPDATE
            """,
            (user_id, expense_id),
        ).fetchone()
        if current is None:
            return "not_found"

        unchanged = (
            current["expense_name"] == expected_original["expense_name"]
            and Decimal(str(current["amount"])) == original_amount
            and int(current["category_id"]) == original_category
            and current["expense_date"] == original_date
            and current["payment_method"] == expected_original["payment_method"]
            and current["notes"] == expected_original["notes"]
        )
        if not unchanged:
            return "changed"

        _ensure_month_category_rows(conn, user_id, category_id, parsed_date.replace(day=1))
        conn.execute(
            """
            UPDATE public.expenses
            SET expense_name = %s, amount = %s, category_id = %s,
                expense_date = %s, payment_method = %s, notes = %s
            WHERE user_id = %s AND expense_id = %s
            """,
            (
                name.strip(), amount_value, category_id, parsed_date,
                payment.strip(), notes, user_id, expense_id,
            ),
        )
        return "updated"


def delete_expense(user_id: str, expense_id: Any) -> bool:
    """Delete an expense belonging to the authenticated user."""
    user_id = _require_user_id(user_id)
    expense_id = _validate_positive_id(expense_id, "Expense ID")
    with user_connection(user_id) as conn:
        result = conn.execute(
            """
            DELETE FROM public.expenses
            WHERE user_id = %s AND expense_id = %s
            """,
            (user_id, expense_id),
        )
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# Category budgets
# ---------------------------------------------------------------------------

def initialise_category_budgets_for_month(user_id: str, budget_month: str) -> None:
    """Create zero allocations for Food and categories active in this month."""
    user_id = _require_user_id(user_id)
    month_start = _parse_budget_month(budget_month)
    with user_connection(user_id) as conn:
        conn.execute(
            """
            INSERT INTO public.category_budgets
                (user_id, category_id, budget_month, budget_amount)
            SELECT c.user_id, c.category_id, %s, 0
            FROM public.categories AS c
            WHERE c.user_id = %s
              AND (
                    c.category_name = 'Food'
                    OR EXISTS (
                        SELECT 1 FROM public.category_periods AS cp
                        WHERE cp.user_id = c.user_id
                          AND cp.category_id = c.category_id
                          AND cp.budget_month = %s
                    )
              )
            ON CONFLICT (category_id, budget_month) DO NOTHING
            """,
            (month_start, user_id, month_start),
        )


def get_category_budgets(user_id: str, budget_month: str) -> dict:
    """Return active category allocations and their summed monthly budget."""
    user_id = _require_user_id(user_id)
    month_start = _parse_budget_month(budget_month)
    with user_connection(user_id) as conn:
        conn.execute(
            """
            INSERT INTO public.category_budgets
                (user_id, category_id, budget_month, budget_amount)
            SELECT c.user_id, c.category_id, %s, 0
            FROM public.categories AS c
            WHERE c.user_id = %s
              AND (
                    c.category_name = 'Food'
                    OR EXISTS (
                        SELECT 1 FROM public.category_periods AS cp
                        WHERE cp.user_id = c.user_id
                          AND cp.category_id = c.category_id
                          AND cp.budget_month = %s
                    )
              )
            ON CONFLICT (category_id, budget_month) DO NOTHING
            """,
            (month_start, user_id, month_start),
        )
        rows = conn.execute(
            """
            SELECT c.category_id, c.category_name, cb.budget_amount
            FROM public.categories AS c
            JOIN public.category_budgets AS cb
              ON cb.category_id = c.category_id
             AND cb.user_id = c.user_id
            WHERE c.user_id = %s
              AND cb.budget_month = %s
              AND (
                    c.category_name = 'Food'
                    OR EXISTS (
                        SELECT 1 FROM public.category_periods AS cp
                        WHERE cp.user_id = c.user_id
                          AND cp.category_id = c.category_id
                          AND cp.budget_month = %s
                    )
              )
            ORDER BY c.category_name
            """,
            (user_id, month_start, month_start),
        ).fetchall()
        categories = [
            {
                "category_id": row["category_id"],
                "category_name": row["category_name"],
                "budget_amount": float(row["budget_amount"]),
            }
            for row in rows
        ]
        return {
            "budget_month": budget_month,
            "categories": categories,
            "total_budget": round(sum(item["budget_amount"] for item in categories), 2),
        }


def get_category_budget(user_id: str, category_id: Any, budget_month: str) -> dict | None:
    """Retrieve one active category allocation for a selected month."""
    user_id = _require_user_id(user_id)
    month_start = _parse_budget_month(budget_month)
    category_id = _validate_positive_id(category_id)
    with user_connection(user_id) as conn:
        row = conn.execute(
            """
            SELECT c.category_id, c.category_name, cb.budget_month, cb.budget_amount
            FROM public.categories AS c
            JOIN public.category_budgets AS cb
              ON cb.category_id = c.category_id AND cb.user_id = c.user_id
            WHERE c.user_id = %s
              AND c.category_id = %s
              AND cb.budget_month = %s
              AND (
                    c.category_name = 'Food'
                    OR EXISTS (
                        SELECT 1 FROM public.category_periods AS cp
                        WHERE cp.user_id = c.user_id
                          AND cp.category_id = c.category_id
                          AND cp.budget_month = %s
                    )
              )
            """,
            (user_id, category_id, month_start, month_start),
        ).fetchone()
        if not row:
            return None
        return {
            "category_id": row["category_id"],
            "category_name": row["category_name"],
            "budget_month": row["budget_month"].strftime("%m-%Y"),
            "budget_amount": float(row["budget_amount"]),
        }


def update_category_budget(
    user_id: str,
    category_id: Any,
    budget_amount: Any,
    budget_month: str,
) -> bool:
    """Set a category's budget for one active month."""
    user_id = _require_user_id(user_id)
    month_start = _parse_budget_month(budget_month)
    category_id = _validate_positive_id(category_id)
    amount_value = _validate_amount(budget_amount, "Budget amount", allow_zero=True)

    with user_connection(user_id) as conn:
        _validate_category_owned(conn, user_id, category_id)
        available = conn.execute(
            """
            SELECT 1
            FROM public.categories AS c
            WHERE c.user_id = %s AND c.category_id = %s
              AND (
                    c.category_name = 'Food'
                    OR EXISTS (
                        SELECT 1 FROM public.category_periods AS cp
                        WHERE cp.user_id = c.user_id
                          AND cp.category_id = c.category_id
                          AND cp.budget_month = %s
                    )
              )
            """,
            (user_id, category_id, month_start),
        ).fetchone()
        if not available:
            raise ValueError("Add this category to the selected month first.")

        conn.execute(
            """
            INSERT INTO public.category_budgets
                (user_id, category_id, budget_month, budget_amount)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (category_id, budget_month)
            DO UPDATE SET budget_amount = EXCLUDED.budget_amount
            """,
            (user_id, category_id, month_start, amount_value),
        )
        return True


def get_monthly_summary(user_id: str, budget_month: str) -> dict:
    """Return monthly spending, total category budget and remaining amount."""
    user_id = _require_user_id(user_id)
    month_start = _parse_budget_month(budget_month)
    with user_connection(user_id) as conn:
        spending_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total_spending
            FROM public.expenses
            WHERE user_id = %s
              AND expense_date >= %s
              AND expense_date < (%s + INTERVAL '1 month')
            """,
            (user_id, month_start, month_start),
        ).fetchone()
        budget_row = conn.execute(
            """
            SELECT COALESCE(SUM(cb.budget_amount), 0) AS total_budget
            FROM public.category_budgets AS cb
            JOIN public.category_periods AS cp
              ON cp.category_id = cb.category_id
             AND cp.user_id = cb.user_id
             AND cp.budget_month = cb.budget_month
            WHERE cb.user_id = %s AND cb.budget_month = %s
            """,
            (user_id, month_start),
        ).fetchone()

        # Food is available by default even without an explicit period row.
        food_budget = conn.execute(
            """
            SELECT COALESCE(SUM(cb.budget_amount), 0) AS food_budget
            FROM public.category_budgets AS cb
            JOIN public.categories AS c
              ON c.category_id = cb.category_id AND c.user_id = cb.user_id
            WHERE cb.user_id = %s
              AND cb.budget_month = %s
              AND c.category_name = 'Food'
              AND NOT EXISTS (
                  SELECT 1 FROM public.category_periods cp
                  WHERE cp.user_id = cb.user_id
                    AND cp.category_id = cb.category_id
                    AND cp.budget_month = cb.budget_month
              )
            """,
            (user_id, month_start),
        ).fetchone()

    spending = float(spending_row["total_spending"])
    budget = float(budget_row["total_budget"]) + float(food_budget["food_budget"])
    remaining = budget - spending
    utilisation = (spending / budget) * 100 if budget > 0 else None
    return {
        "budget_month": budget_month,
        "budget_amount": round(budget, 2),
        "total_spending": round(spending, 2),
        "remaining_budget": round(remaining, 2),
        "utilisation_percentage": round(utilisation, 2) if utilisation is not None else None,
    }


# ---------------------------------------------------------------------------
# Analytics and summaries
# ---------------------------------------------------------------------------

def get_category_summary(
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    category_id: Any = None,
) -> list[dict]:
    """Return category spending with optional date and category filters."""
    user_id = _require_user_id(user_id)
    start, end = _validate_date_range(start_date, end_date)
    category_id = _validate_positive_id(category_id) if category_id is not None else None
    conditions = ["e.user_id = %s"]
    params: list[Any] = [user_id]
    if start:
        conditions.append("e.expense_date >= %s")
        params.append(start)
    if end:
        conditions.append("e.expense_date <= %s")
        params.append(end)
    if category_id is not None:
        conditions.append("e.category_id = %s")
        params.append(category_id)

    with user_connection(user_id) as conn:
        rows = conn.execute(
            f"""
            SELECT c.category_id, c.category_name,
                   SUM(e.amount) AS total_spending,
                   COUNT(e.expense_id) AS transaction_count
            FROM public.expenses AS e
            JOIN public.categories AS c
              ON c.category_id = e.category_id AND c.user_id = e.user_id
            WHERE {' AND '.join(conditions)}
            GROUP BY c.category_id, c.category_name
            ORDER BY total_spending DESC
            """,
            params,
        ).fetchall()

    total = sum(float(row["total_spending"]) for row in rows)
    return [
        {
            "category_id": row["category_id"],
            "category_name": row["category_name"],
            "total_spending": round(float(row["total_spending"]), 2),
            "transaction_count": row["transaction_count"],
            "percentage": round(float(row["total_spending"]) / total * 100, 2) if total else 0,
        }
        for row in rows
    ]


def get_daily_summary(
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    category_id: Any = None,
    payment_method: str | None = None,
) -> list[dict]:
    """Return daily totals with optional date, category and payment filters."""
    user_id = _require_user_id(user_id)
    start, end = _validate_date_range(start_date, end_date)
    conditions = ["e.user_id = %s"]
    params: list[Any] = [user_id]
    if start:
        conditions.append("e.expense_date >= %s")
        params.append(start)
    if end:
        conditions.append("e.expense_date <= %s")
        params.append(end)
    if category_id is not None:
        conditions.append("e.category_id = %s")
        params.append(_validate_positive_id(category_id))
    if payment_method:
        conditions.append("LOWER(e.payment_method) = LOWER(%s)")
        params.append(payment_method.strip())

    with user_connection(user_id) as conn:
        rows = conn.execute(
            f"""
            SELECT e.expense_date, SUM(e.amount) AS total_spending,
                   COUNT(e.expense_id) AS transaction_count
            FROM public.expenses AS e
            WHERE {' AND '.join(conditions)}
            GROUP BY e.expense_date
            ORDER BY e.expense_date
            """,
            params,
        ).fetchall()
    return [
        {
            "expense_date": _display_expense_date(row["expense_date"]),
            "total_spending": round(float(row["total_spending"]), 2),
            "transaction_count": row["transaction_count"],
        }
        for row in rows
    ]


def get_top_expenses(
    user_id: str,
    limit: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    category_id: Any = None,
    payment_method: str | None = None,
) -> list[dict]:
    """Return the largest expenses with optional filters."""
    user_id = _require_user_id(user_id)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer.")
    start, end = _validate_date_range(start_date, end_date)
    conditions = ["e.user_id = %s"]
    params: list[Any] = [user_id]
    if start:
        conditions.append("e.expense_date >= %s")
        params.append(start)
    if end:
        conditions.append("e.expense_date <= %s")
        params.append(end)
    if category_id is not None:
        conditions.append("e.category_id = %s")
        params.append(_validate_positive_id(category_id))
    if payment_method:
        conditions.append("LOWER(e.payment_method) = LOWER(%s)")
        params.append(payment_method.strip())
    params.append(limit)

    with user_connection(user_id) as conn:
        rows = conn.execute(
            f"""
            SELECT e.expense_id, e.expense_name, e.amount, c.category_name,
                   e.expense_date, e.payment_method, e.notes
            FROM public.expenses AS e
            JOIN public.categories AS c
              ON c.category_id = e.category_id AND c.user_id = e.user_id
            WHERE {' AND '.join(conditions)}
            ORDER BY e.amount DESC, e.expense_id DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
    return [
        {
            "expense_id": row["expense_id"],
            "expense_name": row["expense_name"],
            "amount": float(row["amount"]),
            "category_name": row["category_name"],
            "expense_date": _display_expense_date(row["expense_date"]),
            "payment_method": row["payment_method"],
            "notes": row["notes"],
        }
        for row in rows
    ]


def search_expenses(
    user_id: str,
    expense_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    category_id: Any = None,
    payment_method: str | None = None,
) -> list[dict]:
    """Search the user's expenses using optional filters."""
    user_id = _require_user_id(user_id)
    start, end = _validate_date_range(start_date, end_date)
    conditions = ["e.user_id = %s"]
    params: list[Any] = [user_id]
    if expense_name is not None:
        if not isinstance(expense_name, str) or not expense_name.strip():
            raise ValueError("Expense name must be non-empty text.")
        conditions.append("e.expense_name ILIKE %s")
        params.append(f"%{expense_name.strip()}%")
    if start:
        conditions.append("e.expense_date >= %s")
        params.append(start)
    if end:
        conditions.append("e.expense_date <= %s")
        params.append(end)
    if category_id is not None:
        conditions.append("e.category_id = %s")
        params.append(_validate_positive_id(category_id))
    if payment_method:
        conditions.append("LOWER(e.payment_method) = LOWER(%s)")
        params.append(payment_method.strip())

    with user_connection(user_id) as conn:
        rows = conn.execute(
            _expense_select()
            + f"""
            WHERE {' AND '.join(conditions)}
            ORDER BY e.expense_date DESC, e.expense_id DESC
            """,
            params,
        ).fetchall()
    return [_normalise_expense_row(row) for row in rows]


def get_monthly_trend(
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Return monthly spending totals in chronological order."""
    user_id = _require_user_id(user_id)
    start, end = _validate_date_range(start_date, end_date)
    conditions = ["user_id = %s"]
    params: list[Any] = [user_id]
    if start:
        conditions.append("expense_date >= %s")
        params.append(start)
    if end:
        conditions.append("expense_date <= %s")
        params.append(end)

    with user_connection(user_id) as conn:
        rows = conn.execute(
            f"""
            SELECT TO_CHAR(DATE_TRUNC('month', expense_date), 'MM-YYYY') AS month,
                   SUM(amount) AS total_spending,
                   COUNT(expense_id) AS transaction_count
            FROM public.expenses
            WHERE {' AND '.join(conditions)}
            GROUP BY DATE_TRUNC('month', expense_date)
            ORDER BY DATE_TRUNC('month', expense_date)
            """,
            params,
        ).fetchall()
    return [
        {
            "month": row["month"],
            "total_spending": round(float(row["total_spending"]), 2),
            "transaction_count": row["transaction_count"],
        }
        for row in rows
    ]


def get_payment_method_summary(
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    category_id: Any = None,
) -> list[dict]:
    """Return payment-method spending breakdown with optional filters."""
    user_id = _require_user_id(user_id)
    start, end = _validate_date_range(start_date, end_date)
    conditions = ["user_id = %s"]
    params: list[Any] = [user_id]
    if start:
        conditions.append("expense_date >= %s")
        params.append(start)
    if end:
        conditions.append("expense_date <= %s")
        params.append(end)
    if category_id is not None:
        conditions.append("category_id = %s")
        params.append(_validate_positive_id(category_id))

    with user_connection(user_id) as conn:
        rows = conn.execute(
            f"""
            SELECT payment_method, SUM(amount) AS total_spending,
                   COUNT(expense_id) AS transaction_count
            FROM public.expenses
            WHERE {' AND '.join(conditions)}
            GROUP BY payment_method
            ORDER BY total_spending DESC
            """,
            params,
        ).fetchall()
    total = sum(float(row["total_spending"]) for row in rows)
    return [
        {
            "payment_method": row["payment_method"],
            "total_spending": round(float(row["total_spending"]), 2),
            "transaction_count": row["transaction_count"],
            "percentage": round(float(row["total_spending"]) / total * 100, 2) if total else 0,
        }
        for row in rows
    ]


def get_dashboard_summary(
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Return consolidated dashboard metrics for the authenticated user."""
    user_id = _require_user_id(user_id)
    start, end = _validate_date_range(start_date, end_date)
    conditions = ["e.user_id = %s"]
    params: list[Any] = [user_id]
    if start:
        conditions.append("e.expense_date >= %s")
        params.append(start)
    if end:
        conditions.append("e.expense_date <= %s")
        params.append(end)
    where = " AND ".join(conditions)

    with user_connection(user_id) as conn:
        summary = conn.execute(
            f"""
            SELECT COUNT(*) AS transaction_count,
                   COALESCE(SUM(e.amount), 0) AS total_expenses,
                   COALESCE(AVG(e.amount), 0) AS average_expense
            FROM public.expenses AS e
            WHERE {where}
            """,
            params,
        ).fetchone()

        highest_category = conn.execute(
            f"""
            SELECT c.category_name, SUM(e.amount) AS total_spending
            FROM public.expenses AS e
            JOIN public.categories AS c
              ON c.category_id = e.category_id AND c.user_id = e.user_id
            WHERE {where}
            GROUP BY c.category_id, c.category_name
            ORDER BY total_spending DESC
            LIMIT 1
            """,
            params,
        ).fetchone()

        largest_expense = conn.execute(
            f"""
            SELECT e.expense_id, e.expense_name, e.amount,
                   c.category_name, e.expense_date
            FROM public.expenses AS e
            JOIN public.categories AS c
              ON c.category_id = e.category_id AND c.user_id = e.user_id
            WHERE {where}
            ORDER BY e.amount DESC, e.expense_id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()

    return {
        "total_expenses": round(float(summary["total_expenses"]), 2),
        "transaction_count": summary["transaction_count"],
        "average_expense": round(float(summary["average_expense"]), 2),
        "highest_spending_category": (
            {
                "category_name": highest_category["category_name"],
                "total_spending": round(float(highest_category["total_spending"]), 2),
            }
            if highest_category else None
        ),
        "largest_expense": (
            {
                **dict(largest_expense),
                "amount": float(largest_expense["amount"]),
                "expense_date": _display_expense_date(largest_expense["expense_date"]),
            }
            if largest_expense else None
        ),
    }


# Keep direct execution harmless: schema setup is performed in Supabase SQL
# Editor, not by running this module.
if __name__ == "__main__":
    raise SystemExit(
        "FinSight uses Supabase PostgreSQL. Apply the reviewed SQL migration "
        "in Supabase; do not run database.py directly."
    )
