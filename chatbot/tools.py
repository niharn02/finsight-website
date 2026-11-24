
"""
FinSight financial tools.

Provides a controlled interface between the chatbot
and the existing SQLite database and analytics functions.

Read operations expose transaction, budget and analytics data.
Write operations remain subject to the existing graph approval workflow.
"""

from chatbot.scoped_database import (
    add_expense,
    get_categories,
    get_expenses,
    get_expense_by_id,
    search_expenses,
    get_category_budget,
    update_category_budget,
    create_category,
)

from langchain_core.tools import tool

from analytics import (
    get_spending_summary,
    get_average_daily_spending,
    get_weekly_spending,
    get_category_analysis,
    get_payment_method_analysis,
    get_top_expenses_analysis,
    get_category_budget_status,
    get_budget_overview,
    get_monthly_spending_trend,
    get_monthly_spending_comparison,
)


# --------------------------------------------------
# Retrieve available expense categories
# --------------------------------------------------

@tool
def get_categories_tool() -> list:
    """
    Retrieve all expense categories available in FinSight.

    Use this when the user asks about available categories
    or when a category needs to be identified.
    """

    return get_categories()


# --------------------------------------------------
# Retrieve recent expenses
# --------------------------------------------------

@tool
def get_expenses_tool() -> list:
    """
    Retrieve all recorded expenses.

    Use this when the user asks to view their transactions
    without specifying particular search criteria.
    """

    return get_expenses()


# --------------------------------------------------
# Retrieve a specific transaction
# --------------------------------------------------

@tool
def get_expense_details_tool(expense_id: int) -> dict:
    """
    Retrieve the complete details of a transaction by ID.

    Use this when the user provides a specific transaction ID
    or after an exact transaction has been identified.
    """

    # Retrieve the requested expense from the database.
    expense = get_expense_by_id(expense_id)

    # Return a clear result when the transaction does not exist.
    if expense is None:
        return {
            "found": False,
            "message": f"No transaction found with ID {expense_id}."
        }

    return {
        "found": True,
        "expense": expense
    }


# --------------------------------------------------
# Search transactions using available filters
# --------------------------------------------------

@tool
def search_expenses_tool(
    expense_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    category_id: int | None = None,
    payment_method: str | None = None,
) -> list:
    """
    Search expenses using one or more optional filters.

    Dates must use DD-MM-YYYY format.

    Multiple filters can be combined to narrow down results.

    For example:
    - Search by merchant name.
    - Search within a date range.
    - Search by category.
    - Search by payment method.

    Use this to identify the exact transaction before
    attempting an edit or deletion.
    """

    # Execute the existing database search function.
    return search_expenses(
        expense_name=expense_name,
        start_date=start_date,
        end_date=end_date,
        category_id=category_id,
        payment_method=payment_method,
    )


# --------------------------------------------------
# Resolve the exact transaction selected by the user
# --------------------------------------------------

def resolve_transaction_selection(
    matches: list[dict],
    selection: dict | None = None
) -> dict:
    """
    Identify the exact transaction the user wants to modify.

    Supports transaction ID, displayed list number, date and amount.

    This function only identifies a transaction.
    It does not edit or delete any financial record.
    """

    # Return immediately when the search produced no results.
    if not matches:
        return {
            "status": "no_matches",
            "message": "No matching transactions were found.",
            "transaction": None,
            "candidates": []
        }

    # Use an empty selection when the user has not provided one.
    selection = selection or {}

    # Start with all transactions returned by the original search.
    candidates = matches.copy()

    # Track whether the user has provided identifying information.
    has_selection_criteria = False

    # --------------------------------------------------
    # Apply the displayed transaction number
    # --------------------------------------------------

    selection_number = selection.get("selection_number")

    if selection_number is not None:

        # A displayed transaction number must be a positive integer.
        if (
            isinstance(selection_number, bool)
            or not isinstance(selection_number, int)
            or selection_number <= 0
        ):
            return {
                "status": "invalid_selection",
                "message": "Please provide a valid transaction number.",
                "transaction": None,
                "candidates": matches
            }

        # Check that the selected number exists in the displayed list.
        if selection_number > len(candidates):
            return {
                "status": "invalid_selection",
                "message": "That transaction number is not in the list.",
                "transaction": None,
                "candidates": matches
            }

        # Use the original displayed order to identify the selection.
        candidates = [candidates[selection_number - 1]]
        has_selection_criteria = True

    # --------------------------------------------------
    # Apply transaction ID
    # --------------------------------------------------

    transaction_id = selection.get("transaction_id")

    if transaction_id is not None:

        # Match the exact database transaction ID.
        candidates = [
            item for item in candidates
            if item["expense_id"] == transaction_id
        ]

        has_selection_criteria = True

    # --------------------------------------------------
    # Apply expense date
    # --------------------------------------------------

    expense_date = selection.get("expense_date")

    if expense_date is not None:

        # Match the stored date in DD-MM-YYYY format.
        candidates = [
            item for item in candidates
            if item["expense_date"] == expense_date
        ]

        has_selection_criteria = True

    # --------------------------------------------------
    # Apply expense amount
    # --------------------------------------------------

    amount = selection.get("amount")

    if amount is not None:

        # Match the amount using a small tolerance for decimals.
        try:
            selected_amount = float(amount)
        except (ValueError, TypeError):
            return {
                "status": "invalid_selection",
                "message": "Please provide a valid transaction amount.",
                "transaction": None,
                "candidates": matches
            }

        candidates = [
            item for item in candidates
            if abs(float(item["amount"]) - selected_amount) < 0.001
        ]

        has_selection_criteria = True

    # --------------------------------------------------
    # Determine the resolution result
    # --------------------------------------------------

    # No transaction matches the supplied identifying details.
    if not candidates:
        return {
            "status": "not_found",
            "message": "No transaction matches those details.",
            "transaction": None,
            "candidates": []
        }

    # More than one transaction still matches the selection.
    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "message": "Multiple transactions still match. Please clarify.",
            "transaction": None,
            "candidates": candidates
        }

    # Exactly one transaction matches the user's selection criteria.
    if has_selection_criteria:
        return {
            "status": "identified",
            "message": "The requested transaction has been identified.",
            "transaction": candidates[0],
            "candidates": candidates
        }

    # A single search result is unique, but must still be shown to the user.
    return {
        "status": "unique_match",
        "message": "One matching transaction was found. Please verify it.",
        "transaction": candidates[0],
        "candidates": candidates
    }


# Create a new expense in the database.
def create_expense_tool(
    expense_name,
    amount,
    category_id,
    expense_date,
    payment_method,
    notes=None
):
    """
    Create an expense using the existing database function.

    The graph will call this tool only after receiving
    explicit approval from the user.
    """

    try:
        # Insert the expense and retrieve its new ID.
        expense_id = add_expense(
            expense_name=expense_name,
            amount=amount,
            category_id=category_id,
            expense_date=expense_date,
            payment_method=payment_method,
            notes=notes
        )

        # Return a structured result for the chatbot.
        return {
            "success": True,
            "expense_id": expense_id,
            "message": "Expense created successfully."
        }

    except ValueError as error:
        # Return validation errors without exposing database details.
        return {
            "success": False,
            "message": str(error)
        }

    except Exception:
        # Handle unexpected database errors safely.
        return {
            "success": False,
            "message": "Unable to create the expense due to a database error."
        }



# --------------------------------------------------
# Category budget overview tool
# --------------------------------------------------

@tool
def get_budget_overview_tool(budget_month: str) -> dict:
    """Return the overall category-based budget and spending overview."""
    try:
        return {"success": True, **get_budget_overview(budget_month)}
    except ValueError as error:
        return {"success": False, "message": str(error)}
    except Exception:
        return {"success": False, "message": "Unable to retrieve the monthly budget overview."}



# --------------------------------------------------
# Category budget management tools
# --------------------------------------------------

def _find_category_by_name(category_name: str):
    """Resolve a category name without relying on LLM-generated IDs."""
    cleaned_name = str(category_name or "").strip()
    if not cleaned_name:
        raise ValueError("A category name is required.")

    matches = [
        category for category in get_categories()
        if category["category_name"].casefold() == cleaned_name.casefold()
    ]

    if not matches:
        available = ", ".join(item["category_name"] for item in get_categories())
        raise ValueError(
            f"Category '{cleaned_name}' was not found. Available categories: {available}."
        )

    return matches[0]


@tool
def get_category_budgets_tool(budget_month: str) -> dict:
    """Retrieve every category's budget allocation for a month."""
    try:
        from chatbot.scoped_database import get_category_budgets
        return {"success": True, **get_category_budgets(budget_month)}
    except ValueError as error:
        return {"success": False, "message": str(error)}
    except Exception:
        return {"success": False, "message": "Unable to retrieve category budgets."}


@tool
def create_category_budget_tool(
    category_name: str,
    budget_amount: float,
    budget_month: str
) -> dict:
    """Create a category allocation, creating/activating the category if needed.

    This tool is called by the graph only after explicit user approval. The
    database helper creates the master category when it does not exist and
    activates it for the requested month. Existing categories are reused
    case-insensitively, preventing duplicate category records.
    """
    try:
        category_name = str(category_name or "").strip()
        if not category_name:
            raise ValueError("A category name is required.")

        # Create or reuse the category and make it available for this month.
        # This is intentionally done here, after the graph's approval gate.
        category_id = create_category(category_name, budget_month)

        # Retrieve the canonical category name and ID from SQLite.
        category = next(
            (
                item for item in get_categories()
                if int(item["category_id"]) == int(category_id)
            ),
            None
        )
        if category is None:
            raise ValueError("The category could not be retrieved after creation.")

        # Check the existing allocation so this operation never overwrites it.
        current = get_category_budget(category_id, budget_month)
        current_amount = float(current["budget_amount"] or 0) if current else 0.0

        if current_amount > 0:
            return {
                "success": False,
                "already_exists": True,
                "message": (
                    f"A {category['category_name']} budget already exists for "
                    f"{budget_month}: ₹{current_amount:,.2f}. "
                    "Use the edit workflow instead."
                )
            }

        # Save the requested allocation for this category and month.
        update_category_budget(category_id, budget_amount, budget_month)

        return {
            "success": True,
            "category_name": category["category_name"],
            "category_id": int(category_id),
            "budget_amount": float(budget_amount),
            "budget_month": budget_month,
            "message": "Category budget created successfully."
        }
    except ValueError as error:
        return {"success": False, "message": str(error)}
    except Exception:
        return {"success": False, "message": "Unable to create the category budget."}


@tool
def update_category_budget_tool(
    category_name: str,
    budget_amount: float,
    budget_month: str,
    expected_current_amount: float
) -> dict:
    """Update one existing category allocation with a concurrency check."""
    try:
        category = _find_category_by_name(category_name)
        current = get_category_budget(category["category_id"], budget_month)
        if current is None:
            return {"success": False, "message": "No allocation exists for this category and month."}

        current_amount = float(current["budget_amount"] or 0)
        if current_amount <= 0:
            return {"success": False, "message": "No category budget is currently set. Create one instead."}

        if current_amount != float(expected_current_amount):
            return {
                "success": False,
                "changed": True,
                "message": "The category budget changed after the proposal was prepared. Please review it again."
            }

        update_category_budget(category["category_id"], budget_amount, budget_month)
        return {
            "success": True,
            "category_name": category["category_name"],
            "previous_amount": current_amount,
            "budget_amount": float(budget_amount),
            "budget_month": budget_month,
            "message": "Category budget updated successfully."
        }
    except ValueError as error:
        return {"success": False, "message": str(error)}
    except Exception:
        return {"success": False, "message": "Unable to update the category budget."}


@tool
def delete_category_budget_tool(
    category_name: str,
    budget_month: str,
    expected_current_amount: float
) -> dict:
    """Remove an allocation by resetting its amount to zero."""
    try:
        category = _find_category_by_name(category_name)
        current = get_category_budget(category["category_id"], budget_month)
        if current is None:
            return {"success": False, "message": "No category allocation was found for this month."}

        current_amount = float(current["budget_amount"] or 0)
        if current_amount <= 0:
            return {"success": False, "message": f"No {category['category_name']} budget is set for {budget_month}."}

        if current_amount != float(expected_current_amount):
            return {
                "success": False,
                "changed": True,
                "message": "The category budget changed after the deletion proposal was prepared. Please review it again."
            }

        # Keep the category/month row; zero means no allocation is configured.
        update_category_budget(category["category_id"], 0, budget_month)
        return {
            "success": True,
            "category_name": category["category_name"],
            "previous_amount": current_amount,
            "budget_month": budget_month,
            "message": "Category budget removed successfully."
        }
    except ValueError as error:
        return {"success": False, "message": str(error)}
    except Exception:
        return {"success": False, "message": "Unable to remove the category budget."}


# --------------------------------------------------
# Read-only analytics tools
# --------------------------------------------------

@tool
def spending_summary_tool(start_date: str, end_date: str) -> dict:
    """Calculate total spending and transaction count for a date range."""
    return get_spending_summary(start_date, end_date)


@tool
def average_daily_spending_tool(start_date: str, end_date: str) -> dict:
    """Calculate average spending per calendar day for a date range."""
    return get_average_daily_spending(start_date, end_date)


@tool
def weekly_spending_tool(start_date: str, end_date: str) -> dict:
    """Group spending into Monday-to-Sunday weeks within a date range."""
    return get_weekly_spending(start_date, end_date)


@tool
def category_analysis_tool(start_date: str, end_date: str) -> dict:
    """Return category spending totals, transaction counts and percentages."""
    return get_category_analysis(start_date, end_date)


@tool
def payment_method_analysis_tool(start_date: str, end_date: str) -> dict:
    """Return payment-method spending totals and percentage shares."""
    return get_payment_method_analysis(start_date, end_date)


@tool
def top_expenses_analysis_tool(
    start_date: str,
    end_date: str,
    limit: int = 5
) -> dict:
    """Return the largest expenses and their share of period spending."""
    return get_top_expenses_analysis(start_date, end_date, limit=limit)


@tool
def category_budget_status_tool(budget_month: str) -> dict:
    """Compare each category's actual spending with its monthly budget."""
    return get_category_budget_status(budget_month)


@tool
def monthly_spending_trend_tool(
    start_date: str | None = None,
    end_date: str | None = None
) -> dict:
    """Return monthly spending totals and month-over-month changes."""
    return get_monthly_spending_trend(start_date, end_date)


@tool
def monthly_spending_comparison_tool(
    current_month: str,
    previous_month: str,
    category_name: str | None = None
) -> dict:
    """Compare spending between two MM-YYYY months."""
    return get_monthly_spending_comparison(
        current_month, previous_month, category_name
    )
