# Import the Libraries
from flask import Flask, jsonify, request, render_template, send_from_directory, g
import psycopg
import os
from chatbot.scoped_database import (
    get_categories_for_month,
    create_category,
    delete_category_for_month,
    get_category_budgets,
    update_category_budget,
    add_expense,
    get_expenses,
    search_expenses,
    get_expense_by_id,
    update_expense,
    delete_expense,
    get_monthly_summary,
    get_dashboard_summary,
    get_category_summary,
    get_daily_summary,
    get_top_expenses,
    get_payment_method_summary,
    get_monthly_trend
)
import uuid
from uuid import UUID
import traceback
from langchain_core.messages import HumanMessage
from chatbot.graph import build_chatbot_graph
from chatbot.safety import (
    FINANCIAL_ADVICE_REFUSAL,
    contains_financial_recommendation,
    is_financial_advice_request,
)
import calendar
from datetime import datetime, timezone, timedelta
from auth import require_supabase_auth
from database_postgres import initialise_database
from auth_handoff import handoff_bp

# Initialise Flask application
app = Flask(__name__)

# Register the short-lived authentication handoff routes.
app.register_blueprint(handoff_bp)


# Initialise database
initialise_database()


# -------------------------------------------------
# Smart Budget Guidance - Spending Projection
# -------------------------------------------------

# Use Indian Standard Time (UTC+5:30) for budget calculations.
INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


def format_indian_amount(amount):
    """
    Format a number using the Indian numbering system.

    Example:
    1250000 becomes 12,50,000.
    """

    # Round the amount to the nearest whole rupee.
    amount = int(round(float(amount)))

    # Preserve the negative sign, if the amount is negative.
    sign = "-" if amount < 0 else ""

    # Work with the absolute amount.
    amount_text = str(abs(amount))

    # Numbers with three or fewer digits need no additional grouping.
    if len(amount_text) <= 3:
        return sign + amount_text

    # Separate the final three digits from the remaining digits.
    last_three = amount_text[-3:]
    remaining = amount_text[:-3]

    # Group the remaining digits in pairs, as used in Indian numbering.
    groups = []
    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]

    # Combine the grouped digits with the final three digits.
    return sign + ",".join(groups + [last_three])


def calculate_spending_projection(budget_month, monthly_summary):
    """
    Estimate month-end spending using the current average
    daily spending rate.

    Projection formula:
    (Actual spending / Days elapsed) * Total days in month

    Only the current month receives a projection.
    """

    # Get today's date using Indian Standard Time.
    today = datetime.now(INDIA_TIMEZONE).date()

    # Convert the selected budget month from MM-YYYY to a date.
    selected_month = datetime.strptime(
        budget_month, "%m-%Y"
    ).date()

    # Return no projection for a month that is not currently active.
    if (
        selected_month.month != today.month
        or selected_month.year != today.year
    ):
        return {
            "projection_available": False,
            "reason": "Projection is available only for the current month."
        }

    # Calculate the number of calendar days in the current month.
    days_in_month = calendar.monthrange(
        today.year, today.month
    )[1]

    # The current calendar day is included in the elapsed days.
    days_elapsed = today.day

    # Retrieve actual spending from the existing monthly summary.
    actual_spending = float(
        monthly_summary.get("total_spending", 0) or 0
    )

    # Retrieve the monthly budget, if one has been configured.
    budget_amount = float(
        monthly_summary.get("budget_amount", 0) or 0
    )

    # Do not create a misleading projection when no spending exists.
    if actual_spending <= 0:
        return {
            "projection_available": False,
            "reason": (
                "No expenses have been recorded this month yet, "
                "so a spending projection cannot be calculated."
            ),
            "actual_spending": actual_spending,
            "days_elapsed": days_elapsed,
            "days_in_month": days_in_month
        }

    # Calculate the average amount spent per calendar day.
    average_daily_spending = actual_spending / days_elapsed

    # Estimate total spending if the current daily rate continues.
    projected_spending = (
        average_daily_spending * days_in_month
    )

    # Calculate the difference between projected spending and budget.
    projected_difference = projected_spending - budget_amount

    # Generate a human-readable explanation for the user.
    formatted_projection = format_indian_amount(projected_spending)

    explanation = (
        "Based on your spending so far, if you continue spending "
        "at the same rate, your total expenses could reach "
        f"₹{formatted_projection} by the end of this month."
    )

    # Return the projection and the information used to calculate it.
    return {
        "projection_available": True,
        "projection_method": "average_daily_spending",
        "actual_spending": round(actual_spending, 2),
        "average_daily_spending": round(average_daily_spending, 2),
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "projected_month_end_spending": round(projected_spending, 2),
        "monthly_budget": round(budget_amount, 2),
        "projected_difference": round(projected_difference, 2),
        "projected_over_budget": (
            projected_spending > budget_amount
            if budget_amount > 0 else None
        ),
        "explanation": explanation
    }

# -------------------------------------------------
# API Routes
# -------------------------------------------------

# The frontend is a single-page application.
# All page URLs serve the same index.html file; its JavaScript
# handles Dashboard, Transactions, Analytics and Budgets navigation.
@app.route("/")
@app.route("/dashboard.html")
@app.route("/transactions.html")
@app.route("/analytics.html")
@app.route("/budget.html")
def home():
    return send_from_directory(app.template_folder, "index.html")


# Test API connection
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success",
        "message": "FinSight API is running."
    }), 200


# Retrieve all expense categories
@app.route("/api/categories", methods=["GET"])
@require_supabase_auth
def categories():
    try:
        budget_month = request.args.get("month")
        if not budget_month:
            return jsonify({
                "status": "error",
                "message": "Please provide a month (MM-YYYY)."
            }), 400
        categories = get_categories_for_month(budget_month)

        return jsonify({
            "status": "success",
            "count": len(categories),
            "categories": categories
        }), 200

    except ValueError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    except psycopg.Error:
        app.logger.exception("Database error while retrieving categories.")
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve categories."
        }), 500



# Create a category shared by expenses and monthly budgets.
@app.route("/api/categories", methods=["POST"])
@require_supabase_auth
def create_expense_category():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request must contain valid JSON."
            }), 400

        category_name = data.get("category_name")

        if not isinstance(category_name, str) or not category_name.strip():
            return jsonify({
                "status": "error",
                "message": "Please provide a valid category name."
            }), 400

        budget_month = data.get("budget_month")
        if not isinstance(budget_month, str) or not budget_month:
            return jsonify({
                "status": "error",
                "message": "Please provide a budget month (MM-YYYY)."
            }), 400

        category_id = create_category(category_name, budget_month)
        category = next(
            item for item in get_categories_for_month(budget_month)
            if item["category_id"] == category_id
        )

        return jsonify({
            "status": "success",
            "message": "Category created successfully.",
            "category": category
        }), 201

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while creating category.")
        return jsonify({
            "status": "error",
            "message": "Unable to create category."
        }), 500

# Remove a category from the selected month in both Transactions and Budget.
@app.route("/api/categories/<int:category_id>", methods=["DELETE"])
@require_supabase_auth
def remove_category_from_month(category_id):
    try:
        data = request.get_json(silent=True) or {}
        budget_month = data.get("budget_month")
        if not budget_month:
            return jsonify({
                "status": "error",
                "message": "Please provide a budget month (MM-YYYY)."
            }), 400

        delete_category_for_month(category_id, budget_month)
        return jsonify({
            "status": "success",
            "message": "Category removed from the selected month."
        }), 200

    except ValueError as error:
        return jsonify({"status": "error", "message": str(error)}), 409

    except psycopg.Error:
        app.logger.exception("Database error while removing category.")
        return jsonify({
            "status": "error",
            "message": "Unable to remove category."
        }), 500


# Retrieve expenses with optional filters
@app.route("/api/expenses", methods=["GET"])
@require_supabase_auth
def retrieve_expenses():
    try:
        expense_name = request.args.get("expense_name")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        category_id = request.args.get("category_id")
        payment_method = request.args.get("payment_method")

        # Convert category ID to integer
        if category_id is not None:
            try:
                category_id = int(category_id)
            except ValueError:
                return jsonify({
                    "status": "error",
                    "message": "Category ID must be an integer."
                }), 400

        # Retrieve all expenses if no filters are provided
        if not any([
            expense_name,
            start_date,
            end_date,
            category_id is not None,
            payment_method
        ]):
            expenses = get_expenses()
        else:
            expenses = search_expenses(
                expense_name=expense_name,
                start_date=start_date,
                end_date=end_date,
                category_id=category_id,
                payment_method=payment_method
            )

        return jsonify({
            "status": "success",
            "count": len(expenses),
            "expenses": expenses
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while retrieving expenses.")

        return jsonify({
            "status": "error",
            "message": "Unable to retrieve expenses."
        }), 500

# Fetch the Monthly Spending Trend
@app.route("/api/analytics/monthly", methods=["GET"])
@require_supabase_auth
def fetch_monthly_analytics():
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        trend = get_monthly_trend(
            start_date=start_date,
            end_date=end_date
        )

        return jsonify({
            "status": "success",
            "data": trend
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while retrieving monthly analytics.")
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve monthly analytics."
        }), 500

# Retrieve a specific expense by ID
@app.route("/api/expenses/<int:expense_id>", methods=["GET"])
@require_supabase_auth
def retrieve_expense(expense_id):
    try:
        expense = get_expense_by_id(expense_id)

        if expense is None:
            return jsonify({
                "status": "error",
                "message": "Expense not found."
            }), 404

        return jsonify({
            "status": "success",
            "expense": expense
        }), 200

    except psycopg.Error:
        app.logger.exception("Database error while retrieving expense.")

        return jsonify({
            "status": "error",
            "message": "Unable to retrieve expense."
        }), 500


# Add a new expense
@app.route("/api/expenses", methods=["POST"])
@require_supabase_auth
def create_expense():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request must contain valid JSON."
            }), 400

        required_fields = [
            "expense_name",
            "amount",
            "expense_date",
            "payment_method"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data or data[field] is None
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        expense_date = data["expense_date"]
        expense_month = expense_date[3:10] if isinstance(expense_date, str) and len(expense_date) == 10 else None
        category_id = data.get("category_id")
        category_name = data.get("category_name")

        # A typed category is created/activated for the transaction's month.
        if isinstance(category_name, str) and category_name.strip():
            if not expense_month:
                return jsonify({
                    "status": "error",
                    "message": "Please provide a valid expense date."
                }), 400
            category_id = create_category(category_name, expense_month)

        if category_id is None:
            return jsonify({
                "status": "error",
                "message": "Please select or enter a category."
            }), 400

        expense_id = add_expense(
            expense_name=data["expense_name"],
            amount=data["amount"],
            category_id=category_id,
            expense_date=data["expense_date"],
            payment_method=data["payment_method"],
            notes=data.get("notes")
        )

        return jsonify({
            "status": "success",
            "message": "Expense added successfully.",
            "expense_id": expense_id
        }), 201

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while adding expense.")

        return jsonify({
            "status": "error",
            "message": "Unable to add expense."
        }), 500


# Update an existing expense
@app.route("/api/expenses/<int:expense_id>", methods=["PUT"])
@require_supabase_auth
def edit_expense(expense_id):
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request must contain valid JSON."
            }), 400

        # Check whether the expense exists
        existing_expense = get_expense_by_id(expense_id)

        if existing_expense is None:
            return jsonify({
                "status": "error",
                "message": "Expense not found."
            }), 404

        required_fields = [
            "expense_name",
            "amount",
            "category_id",
            "expense_date",
            "payment_method"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data or data[field] is None
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        updated = update_expense(
            expense_id=expense_id,
            expense_name=data["expense_name"],
            amount=data["amount"],
            category_id=data["category_id"],
            expense_date=data["expense_date"],
            payment_method=data["payment_method"],
            notes=data.get("notes")
        )

        if not updated:
            return jsonify({
                "status": "error",
                "message": "Unable to update expense."
            }), 400

        return jsonify({
            "status": "success",
            "message": "Expense updated successfully.",
            "expense_id": expense_id
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while updating expense.")

        return jsonify({
            "status": "error",
            "message": "Unable to update expense."
        }), 500


# Delete an existing expense
@app.route("/api/expenses/<int:expense_id>", methods=["DELETE"])
@require_supabase_auth
def remove_expense(expense_id):
    try:
        # Check whether the expense exists
        existing_expense = get_expense_by_id(expense_id)

        if existing_expense is None:
            return jsonify({
                "status": "error",
                "message": "Expense not found."
            }), 404

        deleted = delete_expense(expense_id)

        if not deleted:
            return jsonify({
                "status": "error",
                "message": "Unable to delete expense."
            }), 400

        return jsonify({
            "status": "success",
            "message": "Expense deleted successfully.",
            "expense_id": expense_id
        }), 200

    except psycopg.Error:
        app.logger.exception("Database error while deleting expense.")

        return jsonify({
            "status": "error",
            "message": "Unable to delete expense."
        }), 500


# Retrieve category-wise allocations and the derived monthly total.
# The first request for a month initialises missing category allocations to zero.
@app.route("/api/budgets/categories", methods=["GET"])
@require_supabase_auth
def fetch_category_budgets():
    try:
        budget_month = request.args.get("month")

        if not budget_month:
            return jsonify({
                "status": "error",
                "message": "Please provide a budget month (MM-YYYY)."
            }), 400

        budget_data = get_category_budgets(budget_month)

        return jsonify({
            "status": "success",
            "data": budget_data
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception(
            "Database error while retrieving category budgets."
        )
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve category budgets."
        }), 500


# Create or update one category's allocation for a selected month.
@app.route("/api/budgets/categories", methods=["PUT"])
@require_supabase_auth
def save_category_budget():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request must contain valid JSON."
            }), 400

        required_fields = ["category_id", "budget_amount", "budget_month"]
        missing_fields = [
            field for field in required_fields
            if field not in data or data[field] is None
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        if isinstance(data["category_id"], bool):
            return jsonify({
                "status": "error",
                "message": "Category ID must be a valid integer."
            }), 400

        try:
            category_id = int(data["category_id"])
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Category ID must be a valid integer."
            }), 400

        update_category_budget(
            category_id=category_id,
            budget_amount=data["budget_amount"],
            budget_month=data["budget_month"]
        )

        # Return the updated category allocations and freshly derived total.
        budget_data = get_category_budgets(data["budget_month"])

        return jsonify({
            "status": "success",
            "message": "Category budget saved successfully.",
            "data": budget_data
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while saving category budget.")
        return jsonify({
            "status": "error",
            "message": "Unable to save category budget."
        }), 500


# Fetch the complete budget summary, including category spending
# and the current month's spending projection.
@app.route("/api/budget/summary", methods=["GET"])
@require_supabase_auth
def fetch_budget_summary():
    try:
        # Retrieve the selected month in MM-YYYY format.
        budget_month = request.args.get("month")

        # Stop if the month was not supplied.
        if not budget_month:
            return jsonify({
                "status": "error",
                "message": "Please provide a budget month (MM-YYYY)."
            }), 400

        # Convert the selected month into a date object.
        selected_month = datetime.strptime(
            budget_month,
            "%m-%Y"
        )

        # Determine the first day of the selected month.
        start_date = selected_month.replace(day=1)

        # Determine the last day of the selected month.
        days_in_month = calendar.monthrange(
            selected_month.year,
            selected_month.month
        )[1]

        end_date = selected_month.replace(
            day=days_in_month
        )

        # Convert dates into the DD-MM-YYYY format expected
        # by the existing category summary function.
        start_date_str = start_date.strftime("%d-%m-%Y")
        end_date_str = end_date.strftime("%d-%m-%Y")

        # Retrieve the existing monthly budget and spending figures.
        summary = get_monthly_summary(budget_month)

        # The monthly budget is derived from category allocations.
        # No separate legacy overall-budget record is required.
        budget_exists = float(summary.get("budget_amount", 0) or 0) > 0

        # Retrieve category-wise spending for the selected month.
        categories = get_category_summary(
            start_date=start_date_str,
            end_date=end_date_str
        )

        # Calculate the spending projection.
        # The projection function itself checks whether this
        # is the current month before producing an estimate.
        projection = calculate_spending_projection(
            budget_month,
            summary
        )

        # Preserve all existing summary fields at the top level.
        # This keeps the current frontend compatible.
        response_data = {
            **summary,
            "budget_exists": budget_exists,
            "categories": categories,
            "projection": projection
        }

        return jsonify({
            "status": "success",
            "data": response_data
        }), 200

    except ValueError as error:
        # Return invalid month or date errors to the frontend.
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        # Log database errors without exposing technical details.
        app.logger.exception(
            "Database error while retrieving budget summary."
        )

        return jsonify({
            "status": "error",
            "message": "Unable to retrieve budget summary."
        }), 500



# Fetch the Analytics Dashboard
@app.route("/api/analytics/dashboard", methods=["GET"])
@require_supabase_auth
def fetch_dashboard_analytics():
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        summary = get_dashboard_summary(
            start_date=start_date,
            end_date=end_date
        )

        return jsonify({
            "status": "success",
            "data": summary
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while retrieving dashboard analytics.")
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve dashboard analytics."
        }), 500

# Fetch the Category Analytics
@app.route("/api/analytics/categories", methods=["GET"])
@require_supabase_auth
def fetch_category_analytics():
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        summary = get_category_summary(
            start_date=start_date,
            end_date=end_date
        )

        return jsonify({
            "status": "success",
            "data": summary
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while retrieving category analytics.")
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve category analytics."
        }), 500

# Get Daily Summary
@app.route("/api/analytics/daily", methods=["GET"])
@require_supabase_auth
def fetch_daily_analytics():
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        category_id = request.args.get("category_id")
        payment_method = request.args.get("payment_method")

        if category_id:
            try:
                category_id = int(category_id)
            except ValueError:
                return jsonify({
                    "status": "error",
                    "message": "Category ID must be a valid integer."
                }), 400

        summary = get_daily_summary(
            start_date=start_date,
            end_date=end_date,
            category_id=category_id,
            payment_method=payment_method
        )

        return jsonify({
            "status": "success",
            "data": summary
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while retrieving daily analytics.")
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve daily analytics."
        }), 500

# Get the Top Expenses
@app.route("/api/analytics/top-expenses", methods=["GET"])
@require_supabase_auth
def fetch_top_expenses():
    try:
        limit = request.args.get("limit", default=5)

        try:
            limit = int(limit)
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Limit must be a valid integer."
            }), 400

        if limit < 1 or limit > 100:
            return jsonify({
                "status": "error",
                "message": "Limit must be between 1 and 100."
            }), 400

        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        category_id = request.args.get("category_id")
        payment_method = request.args.get("payment_method")

        if category_id:
            try:
                category_id = int(category_id)
            except ValueError:
                return jsonify({
                    "status": "error",
                    "message": "Category ID must be a valid integer."
                }), 400

        expenses = get_top_expenses(
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            category_id=category_id,
            payment_method=payment_method
        )

        return jsonify({
            "status": "success",
            "data": expenses
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception("Database error while retrieving top expenses.")
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve top expenses."
        }), 500

# Fetch the Payment Method Summary
@app.route("/api/analytics/payment-methods", methods=["GET"])
@require_supabase_auth
def fetch_payment_method_analytics():
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        summary = get_payment_method_summary(
            start_date=start_date,
            end_date=end_date
        )

        return jsonify({
            "status": "success",
            "data": summary
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except psycopg.Error:
        app.logger.exception(
            "Database error while retrieving payment method analytics."
        )
        return jsonify({
            "status": "error",
            "message": "Unable to retrieve payment method analytics."
        }), 500


# -------------------------------------------------
# Chatbot API
# -------------------------------------------------

@app.route("/api/chat", methods=["POST"])
@require_supabase_auth
def chat():
    """
    Receive a chatbot message, invoke the LangGraph workflow,
    and return the assistant response with the persistent session ID.
    """

    print("[DEBUG] /api/chat route was reached", flush=True)
    try:
        # 1. Read and validate the incoming JSON request
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            app.logger.warning(
                "Chat request rejected: body is not valid JSON. Content-Type: %s",
                request.content_type
            )
            return jsonify({
                "status": "error",
                "message": "Request must contain valid JSON."
            }), 400

        message = data.get("message")
        session_id = data.get("session_id")

        if not isinstance(message, str) or not message.strip():
            app.logger.warning("Chat request rejected: empty or non-text message.")
            return jsonify({
                "status": "error",
                "message": "Please enter a message."
            }), 400

        # Reuse the existing session ID or create a new one.
        if not session_id:
            session_id = str(uuid.uuid4())

        if not isinstance(session_id, str) or len(session_id) > 128:
            app.logger.warning("Chat request rejected: invalid session ID.")
            return jsonify({
                "status": "error",
                "message": "Invalid session ID."
            }), 400

        try:
            # A browser session ID is only an opaque UUID, never a global thread key.
            session_id = str(UUID(session_id))
        except (ValueError, TypeError, AttributeError):
            return jsonify({"status": "error", "message": "Invalid session ID."}), 400

        thread_id = f"{g.user_id}:{session_id}"

        # Limit each conversation to ten user messages.
        # The server enforces this limit so changing frontend code or browser
        # storage cannot extend a conversation beyond the configured limit.
        max_user_turns = 10
        previous_user_turns = 0
        conversation_reset = False

        # 2. Build and invoke the LangGraph workflow.
        try:
            chatbot_graph = build_chatbot_graph()
            thread_config = {"configurable": {"thread_id": thread_id}}

            # Check the existing checkpoint before adding the next message.
            # A HumanMessage represents one user interaction; assistant and
            # internal tool messages do not increase this counter.
            if session_id:
                checkpoint = chatbot_graph.get_state(thread_config)
                checkpoint_values = getattr(checkpoint, "values", {}) or {}
                previous_messages = checkpoint_values.get("messages", [])

                previous_user_turns = sum(
                    1 for item in previous_messages
                    if getattr(item, "type", "") == "human"
                )

                # Start a new LangGraph thread when the previous conversation
                # already contains ten user messages. The current request then
                # becomes the first message in the new conversation.
                if previous_user_turns >= max_user_turns:
                    session_id = str(uuid.uuid4())
                    conversation_reset = True
                    thread_id = f"{g.user_id}:{session_id}"
                    thread_config = {
                        "configurable": {"thread_id": thread_id}
                    }
                    previous_user_turns = 0

            app.logger.info(
                "Starting chatbot workflow. Session: %s; previous user turns: %s",
                session_id,
                previous_user_turns
            )

            print("[DEBUG] About to invoke LangGraph", flush=True)

            result = chatbot_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]},
                config=thread_config
            )

            print("[DEBUG] LangGraph invocation completed", flush=True)
            # Log the result structure only, not the financial message content
            app.logger.info(
                "Chatbot workflow completed. Result type: %s, messages: %s",
                type(result).__name__,
                len(result.get("messages", [])) if isinstance(result, dict) else 0
            )

        except Exception:
            app.logger.exception("LangGraph workflow execution failed. Session: %s", session_id)
            raise

        # 3. Extract the latest assistant response
        assistant_response = ""
        messages = result.get("messages", []) if isinstance(result, dict) else []

        # Search messages from newest to oldest
        for msg in reversed(messages):
            if getattr(msg, "type", None) != "ai":
                continue

            content = getattr(msg, "content", None)

            # Normal text response
            if isinstance(content, str) and content.strip():
                assistant_response = content.strip()
                break

            # Structured content blocks
            if isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                assistant_response = "\n".join(p for p in text_parts if p).strip()

                if assistant_response:
                    break

        # Use the response field only if no AI message was found
        if not assistant_response and isinstance(result, dict):
            fallback_response = result.get("response", "")

            if isinstance(fallback_response, str):
                assistant_response = fallback_response.strip()

        # Final API-level safety boundary: never return personalised financial
        # or investment recommendations, even if an upstream graph node fails
        # to apply its own policy.
        if (
            is_financial_advice_request(message)
            or contains_financial_recommendation(assistant_response)
        ):
            app.logger.warning(
                "Financial-advice boundary applied to chatbot response. Session: %s",
                session_id
            )
            assistant_response = FINANCIAL_ADVICE_REFUSAL

        # Warn only when the workflow produced no usable answer
        if not assistant_response:
            app.logger.warning(
                "Chatbot returned an empty response. Session: %s", session_id
            )
            return jsonify({
                "status": "error",
                "message": "The chatbot did not return a response. Please try again.",
                "session_id": session_id
            }), 500

        # 4. Return the assistant response to the frontend
        app.logger.info("Chat response sent. Session: %s", session_id)
        return jsonify({
            "status": "success",
            "response": assistant_response,
            "session_id": session_id,
            "conversation_reset": conversation_reset,
            "turn_count": previous_user_turns + 1,
            "turn_limit": max_user_turns,
            # Notify the frontend after the ninth user message so it can warn
            # that one final message remains in this conversation.
            "show_turn_limit_warning": previous_user_turns + 1 == max_user_turns - 1
        }), 200

    except Exception as e:
        
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/chat/history", methods=["GET"])
@require_supabase_auth
def chat_history():
    """Return the visible human/assistant messages for a LangGraph thread."""
    try:
        session_id = request.args.get("session_id", "")

        if not session_id or not isinstance(session_id, str) or len(session_id) > 128:
            return jsonify({
                "status": "error",
                "message": "A valid session ID is required."
            }), 400
        try:
            session_id = str(UUID(session_id))
        except (ValueError, TypeError, AttributeError):
            return jsonify({"status": "error", "message": "Invalid session ID."}), 400

        # Namespace the checkpoint by verified identity to prevent cross-user reads.
        thread_id = f"{g.user_id}:{session_id}"
        chatbot_graph = build_chatbot_graph()
        state = chatbot_graph.get_state({
            "configurable": {"thread_id": thread_id}
        })
        state_values = getattr(state, "values", {}) or {}
        graph_messages = state_values.get("messages", [])

        history = []
        for item in graph_messages:
            message_type = getattr(item, "type", "")
            if message_type not in ("human", "ai"):
                continue

            content = getattr(item, "content", "")
            if isinstance(content, list):
                # Convert text content blocks into a simple displayable string.
                content = "\\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            elif not isinstance(content, str):
                content = str(content)

            # Skip empty AI messages used only for tool calls.
            if not content.strip():
                continue

            history.append({
                "role": "user" if message_type == "human" else "assistant",
                "content": content
            })

        return jsonify({
            "status": "success",
            "messages": history,
            "session_id": session_id
        }), 200

    except Exception:
        
        app.logger.exception("Unable to retrieve chatbot conversation history.")
        return jsonify({
            "status": "error",
            "message": "Unable to load this conversation. Please try again."
        }), 500


# -------------------------------------------------
# Run Flask Application
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
