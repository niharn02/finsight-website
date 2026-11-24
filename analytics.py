
# Import database functions required for monthly spending analytics.
from chatbot.scoped_database import (
    get_category_summary,
    get_top_expenses,
    search_expenses,
    get_daily_summary,
    get_monthly_trend,
    get_payment_method_summary,
    get_category_budgets,
    get_monthly_summary,
)

# Import date utilities for calculating comparable periods.
from datetime import datetime
import calendar
from zoneinfo import ZoneInfo


def compare_monthly_spending(
    current_month,
    previous_month,
    category_name=None
):
    """
    Compare spending between two months.

    Expected month format: MM-YYYY
    Example: 11-2026

    If the current month is still ongoing, compare both months
    using the same number of elapsed calendar days.
    """

    # Convert month strings into date ranges.
    def get_month_dates(month, comparison_day=None):

        # Convert the supplied month into a date object.
        month_date = datetime.strptime(month, "%m-%Y")

        # Find the number of days in the requested month.
        last_day = calendar.monthrange(
            month_date.year,
            month_date.month
        )[1]

        # Use the full month unless a comparison day is supplied.
        end_day = (
            min(comparison_day, last_day)
            if comparison_day is not None
            else last_day
        )

        # Build the first and last dates in DD-MM-YYYY format.
        start_date = month_date.replace(day=1).strftime(
            "%d-%m-%Y"
        )

        end_date = month_date.replace(day=end_day).strftime(
            "%d-%m-%Y"
        )

        return start_date, end_date

    # Get today's date using the Indian timezone.
    today = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).date()

    # Check whether the current comparison month is ongoing.
    is_current_month = (
        current_month == today.strftime("%m-%Y")
    )

    # Compare equivalent elapsed periods for an ongoing month.
    comparison_day = today.day if is_current_month else None

    
    # Retrieve the current month's date range.
    # If it is the ongoing month, use the first day through today.
    current_start, current_end = get_month_dates(
        current_month,
        comparison_day
    )

    # Retrieve the complete previous month's date range.
    # Do not restrict it to the same number of elapsed days.
    previous_start, previous_end = get_month_dates(
        previous_month
    )

    # Resolve the requested category name to its database ID.
    category_id = None

    if category_name:
        # Retrieve category records to find the matching category.
        all_categories = get_category_summary()

        matching_category = next(
            (
                category
                for category in all_categories
                if category["category_name"].strip().casefold()
                == category_name.strip().casefold()
            ),
            None
        )

        # Stop if the requested category does not exist.
        if matching_category is None:
            raise ValueError(
                f"Category '{category_name}' was not found."
            )

        category_id = matching_category["category_id"]

    # Fetch category-wise spending for both date ranges.
    current_categories = get_category_summary(
        start_date=current_start,
        end_date=current_end,
        category_id=category_id,
    )

    # Retrieve only the requested category's spending
    # from the previous comparison period.
    previous_categories = get_category_summary(
        start_date=previous_start,
        end_date=previous_end,
        category_id=category_id
    )

    # Calculate total spending from the filtered category results.
    current_total = sum(
        float(category["total_spending"] or 0)
        for category in current_categories
    )

    previous_total = sum(
        float(category["total_spending"] or 0)
        for category in previous_categories
    )

    # Calculate the absolute and percentage difference.
    spending_difference = current_total - previous_total

    if previous_total > 0:
        percentage_change = (
            spending_difference / previous_total
        ) * 100
    else:
        # Percentage change cannot be calculated from a zero baseline.
        percentage_change = None

    # Fetch the five largest transactions for both periods.
    current_top_expenses = get_top_expenses(
        limit=5,
        start_date=current_start,
        end_date=current_end,
        category_id=category_id,
    )

    previous_top_expenses = get_top_expenses(
        limit=5,
        start_date=previous_start,
        end_date=previous_end,
        category_id=category_id,
    )


    # --------------------------------------------------
    # Retrieve all transactions for detailed analysis.
    # --------------------------------------------------

    # Fetch only transactions belonging to the requested category
    # during the current comparison period.
    current_transactions = search_expenses(
        start_date=current_start,
        end_date=current_end,
        category_id=category_id
    )

    # Fetch only transactions belonging to the requested category
    # during the previous comparison period.
    previous_transactions = search_expenses(
        start_date=previous_start,
        end_date=previous_end,
        category_id=category_id
    )

    # Calculate the combined value of the five largest expenses.
    current_top_expense_total = sum(
        float(expense["amount"] or 0)
        for expense in current_top_expenses
    )

    previous_top_expense_total = sum(
        float(expense["amount"] or 0)
        for expense in previous_top_expenses
    )

    # Calculate what percentage of period spending these expenses represent.
    current_top_expense_percentage = (
        (current_top_expense_total / current_total) * 100
        if current_total > 0 else 0
    )

    previous_top_expense_percentage = (
        (previous_top_expense_total / previous_total) * 100
        if previous_total > 0 else 0
    )

    # Create dictionaries for quick category-wise comparison.
    current_category_map = {
        category["category_name"]: float(
            category["total_spending"] or 0
        )
        for category in current_categories
    }

    previous_category_map = {
        category["category_name"]: float(
            category["total_spending"] or 0
        )
        for category in previous_categories
    }

    # Combine category names from both months.
    all_categories = (
        set(current_category_map)
        | set(previous_category_map)
    )

    category_changes = []

    # Calculate the spending difference for each category.
    for category_name in all_categories:

        # Retrieve spending for each month.
        current_amount = current_category_map.get(
            category_name, 0
        )

        previous_amount = previous_category_map.get(
            category_name, 0
        )

        # Calculate the difference between the two periods.
        difference = current_amount - previous_amount

        # Determine whether spending increased, decreased or stayed unchanged.
        if difference > 0:
            status = "increased"
        elif difference < 0:
            status = "decreased"
        else:
            status = "unchanged"

        # Store the category comparison.
        category_changes.append({
            "category_name": category_name,
            "current_spending": current_amount,
            "previous_spending": previous_amount,
            "difference": difference,
            "status": status
        })

    # Sort categories by the absolute size of their spending change.
    category_changes.sort(
        key=lambda item: abs(item["difference"]),
        reverse=True
    )

    # Return all comparison data in one structured result.
    return {
        "current_month": current_month,
        "previous_month": previous_month,

        # Comparison period information.
        "current_start_date": current_start,
        "current_end_date": current_end,
        "previous_start_date": previous_start,
        "previous_end_date": previous_end,

        # Overall spending comparison.
        "current_total": current_total,
        "previous_total": previous_total,
        "spending_difference": spending_difference,
        "percentage_change": percentage_change,

        # Category-wise spending.
        "current_categories": current_categories,
        "previous_categories": previous_categories,
        "category_changes": category_changes,

        "category_name": category_name,

        # Largest transactions.
        "current_top_expenses": current_top_expenses,
        "previous_top_expenses": previous_top_expenses,

        # All individual transactions for detailed analysis.
        "current_transactions": current_transactions,
        "previous_transactions": previous_transactions,

        # Largest transaction totals and their contribution.
        "current_top_expense_total": current_top_expense_total,
        "previous_top_expense_total": previous_top_expense_total,
        "current_top_expense_percentage": current_top_expense_percentage,
        "previous_top_expense_percentage": previous_top_expense_percentage
    }



def generate_monthly_comparison_insight(result):
    """
    Build a friendly, easy-to-scan monthly spending comparison.

    The figures and transaction groups come directly from the
    comparison result; this function only controls presentation.
    """

    # Retrieve the verified comparison figures.
    current_total = float(result["current_total"] or 0)
    previous_total = float(result["previous_total"] or 0)
    difference = float(result["spending_difference"] or 0)
    percentage = result["percentage_change"]

    # Convert MM-YYYY into readable month names.
    current_month_label = datetime.strptime(
        result["current_month"], "%m-%Y"
    ).strftime("%B %Y")

    previous_month_label = datetime.strptime(
        result["previous_month"], "%m-%Y"
    ).strftime("%B %Y")

    insights = [
        "## 📊 Your monthly spending snapshot",
        ""
    ]

    # --------------------------------------------------
    # 1. Overall spending summary.
    # --------------------------------------------------

    if difference < 0:
        summary = (
            f"**Good progress!** Your spending was lower by "
            f"**₹{abs(difference):,.2f}**"
        )

        if percentage is not None:
            summary += f" (**{abs(percentage):.2f}%**)"

        summary += " compared with the previous period."

    elif difference > 0:
        summary = (
            f"Your spending was higher by **₹{difference:,.2f}**"
        )

        if percentage is not None:
            summary += f" (**{percentage:.2f}%**)"

        summary += " compared with the previous period."

    else:
        summary = (
            "**Your spending remained unchanged** compared "
            "with the previous period."
        )

    insights.append(summary)
    insights.append("")
    insights.append(
        f"- **{current_month_label}:** ₹{current_total:,.2f}"
    )
    insights.append(
        f"- **{previous_month_label}:** ₹{previous_total:,.2f}"
    )

    # Return a concise comparison instead of adding lengthy
    # category changes and transaction-level details.
    return "\n".join(insights)


    # --------------------------------------------------
    # 2. Category-wise contribution to spending changes.
    # --------------------------------------------------

    category_changes = result.get("category_changes", [])

    # Sort categories by the absolute size of their spending change.
    sorted_categories = sorted(
        category_changes,
        key=lambda item: abs(float(item.get("difference", 0) or 0)),
        reverse=True
    )

    category_insights = []

    for category in sorted_categories:
        name = category["category_name"]
        current = float(category["current_spending"] or 0)
        previous = float(category["previous_spending"] or 0)
        change = current - previous

        # Skip categories with no change.
        if change == 0:
            continue

        if change < 0:
            category_insights.append(
                f"- **{name}:** decreased by ₹{abs(change):,.2f} "
                f"(₹{previous:,.2f} → ₹{current:,.2f})"
            )
        else:
            category_insights.append(
                f"- **{name}:** increased by ₹{change:,.2f} "
                f"(₹{previous:,.2f} → ₹{current:,.2f})"
            )

    if category_insights:
        insights.extend([
            "",
            "### 🧭 What changed by category",
            *category_insights
        ])
    else:
        insights.extend([
            "",
            "### 🧭 What changed by category",
            "- No category-level spending changes were recorded."
        ])

    # --------------------------------------------------
    # 3. Largest individual transaction groups.
    # --------------------------------------------------

    def format_top_transactions(transactions, limit=3):
        """
        Group transactions with the same expense name and category,
        then return the largest groups by their combined amount.
        """

        grouped = {}

        for item in transactions:
            # Use name and category together to avoid mixing
            # unrelated expenses that share the same name.
            name = item.get("expense_name", "Expense")
            category = item.get("category_name", "Uncategorised")
            key = (name.strip().lower(), category.strip().lower())

            if key not in grouped:
                grouped[key] = {
                    "expense_name": name,
                    "category_name": category,
                    "amount": 0,
                    "count": 0
                }

            # Accumulate the amount and number of matching transactions.
            grouped[key]["amount"] += float(item.get("amount", 0) or 0)
            grouped[key]["count"] += 1

        # Keep the largest groups first.
        largest_groups = sorted(
            grouped.values(),
            key=lambda item: item["amount"],
            reverse=True
        )[:limit]

        formatted = []

        for item in largest_groups:
            count_label = (
                f" × {item['count']}"
                if item["count"] > 1
                else ""
            )

            formatted.append(
                f"- **{item['expense_name']}{count_label}** "
                f"({item['category_name']}) — ₹{item['amount']:,.2f}"
            )

        return formatted

    current_transactions = result.get("current_transactions", [])
    previous_transactions = result.get("previous_transactions", [])

    current_largest = format_top_transactions(current_transactions)
    previous_largest = format_top_transactions(previous_transactions)

    if current_largest or previous_largest:
        insights.extend([
            "",

            "### 🧾 Your biggest spending items"
        
        ])

        if current_largest:
            insights.extend([
                f"**{current_month_label}**",
                *current_largest
            ])

        if previous_largest:
            insights.extend([
                f"**{previous_month_label}**",
                *previous_largest
            ])

    # --------------------------------------------------
    # 4. Comparison period details.
    # --------------------------------------------------

    insights.extend([
        "",

        "### 📅 Period covered",
        
        (
            f"- **{previous_month_label}:** "
            f"{result['previous_start_date']} to "
            f"{result['previous_end_date']}"
        ),
        (
            f"- **{current_month_label}:** "
            f"{result['current_start_date']} to "
            f"{result['current_end_date']}"
        )
    ])

    # Return Markdown so the chatbot can display headings and lists.
    return "\\n".join(insights)


# --------------------------------------------------
# Reusable analytics functions
# --------------------------------------------------

from datetime import date, timedelta


def _validate_period(start_date, end_date):
    """Validate and parse the DD-MM-YYYY date range used by the database."""
    if not start_date or not end_date:
        raise ValueError("Both start_date and end_date are required.")

    start = datetime.strptime(start_date, "%d-%m-%Y").date()
    end = datetime.strptime(end_date, "%d-%m-%Y").date()

    if start > end:
        raise ValueError("Start date cannot be after end date.")

    return start, end


def get_spending_summary(start_date, end_date, category_id=None,
                         payment_method=None):
    """Return total spending, transaction count and active spending days."""
    _validate_period(start_date, end_date)

    daily_rows = get_daily_summary(
        start_date=start_date,
        end_date=end_date,
        category_id=category_id,
        payment_method=payment_method
    )

    total = sum(float(row.get("total_spending") or 0) for row in daily_rows)
    count = sum(int(row.get("transaction_count") or 0) for row in daily_rows)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_spending": round(total, 2),
        "transaction_count": count,
        "spending_days": len(daily_rows)
    }


def get_average_daily_spending(start_date, end_date, category_id=None,
                               payment_method=None):
    """Calculate the daily average across every calendar day in the period."""
    start, end = _validate_period(start_date, end_date)
    summary = get_spending_summary(
        start_date, end_date, category_id, payment_method
    )
    calendar_days = (end - start).days + 1

    return {
        **summary,
        "calendar_days": calendar_days,
        "average_daily_spending": round(
            summary["total_spending"] / calendar_days, 2
        ) if calendar_days else 0.0
    }


def get_weekly_spending(start_date, end_date, category_id=None,
                        payment_method=None):
    """Group daily totals into Monday-to-Sunday weeks, clipped to the period."""
    start, end = _validate_period(start_date, end_date)
    daily_rows = get_daily_summary(
        start_date=start_date,
        end_date=end_date,
        category_id=category_id,
        payment_method=payment_method
    )

    daily_amounts = {}
    for row in daily_rows:
        day = datetime.strptime(row["expense_date"], "%d-%m-%Y").date()
        daily_amounts[day] = float(row.get("total_spending") or 0)

    weeks = []
    week_start = start - timedelta(days=start.weekday())

    while week_start <= end:
        clipped_start = max(week_start, start)
        clipped_end = min(week_start + timedelta(days=6), end)
        amount = sum(
            value for day, value in daily_amounts.items()
            if clipped_start <= day <= clipped_end
        )
        weeks.append({
            "start_date": clipped_start.strftime("%d-%m-%Y"),
            "end_date": clipped_end.strftime("%d-%m-%Y"),
            "total_spending": round(amount, 2)
        })
        week_start += timedelta(days=7)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "weeks": weeks,
        "total_spending": round(sum(w["total_spending"] for w in weeks), 2)
    }


def get_category_analysis(start_date, end_date, category_id=None):
    """Return category totals with transaction counts and spending shares."""
    _validate_period(start_date, end_date)
    rows = get_category_summary(
        start_date=start_date,
        end_date=end_date,
        category_id=category_id
    )
    total = sum(float(row.get("total_spending") or 0) for row in rows)

    categories = []
    for row in rows:
        amount = float(row.get("total_spending") or 0)
        categories.append({
            **row,
            "total_spending": round(amount, 2),
            "percentage": round(amount / total * 100, 2) if total else 0.0
        })

    categories.sort(key=lambda item: item["total_spending"], reverse=True)
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_spending": round(total, 2),
        "categories": categories
    }


def get_payment_method_analysis(start_date, end_date, category_id=None):
    """Return payment-method totals with transaction counts and shares."""
    _validate_period(start_date, end_date)
    rows = get_payment_method_summary(
        start_date=start_date,
        end_date=end_date,
        category_id=category_id
    )
    total = sum(float(row.get("total_spending") or 0) for row in rows)

    methods = []
    for row in rows:
        amount = float(row.get("total_spending") or 0)
        methods.append({
            **row,
            "total_spending": round(amount, 2),
            "percentage": round(amount / total * 100, 2) if total else 0.0
        })

    methods.sort(key=lambda item: item["total_spending"], reverse=True)
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_spending": round(total, 2),
        "payment_methods": methods
    }


def get_top_expenses_analysis(start_date, end_date, limit=5,
                              category_id=None, payment_method=None):
    """Return the largest individual expenses and their share of spending."""
    _validate_period(start_date, end_date)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    expenses = get_top_expenses(
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        category_id=category_id,
        payment_method=payment_method
    )
    summary = get_spending_summary(
        start_date, end_date, category_id, payment_method
    )
    top_total = sum(float(item.get("amount") or 0) for item in expenses)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "expenses": expenses,
        "top_expenses_total": round(top_total, 2),
        "total_spending": summary["total_spending"],
        "percentage_of_spending": round(
            top_total / summary["total_spending"] * 100, 2
        ) if summary["total_spending"] else 0.0
    }


def get_category_budget_status(budget_month):
    """Compare each active category budget with its actual month spending."""
    month_date = datetime.strptime(budget_month, "%m-%Y")
    last_day = calendar.monthrange(month_date.year, month_date.month)[1]
    start_date = month_date.replace(day=1).strftime("%d-%m-%Y")
    end_date = month_date.replace(day=last_day).strftime("%d-%m-%Y")

    budget_data = get_category_budgets(budget_month)
    spending_rows = get_category_summary(
        start_date=start_date,
        end_date=end_date
    )
    spending_by_id = {
        row["category_id"]: float(row.get("total_spending") or 0)
        for row in spending_rows
    }

    categories = []
    for category in budget_data.get("categories", []):
        budget = float(category.get("budget_amount") or 0)
        spent = spending_by_id.get(category["category_id"], 0.0)
        remaining = budget - spent
        utilisation = spent / budget * 100 if budget > 0 else None

        # Keep status thresholds consistent with the FinSight budget rules.
        if budget <= 0:
            status = "no_budget"
        elif spent > budget + 0.005:
            status = "overspent"
        elif spent >= budget - 0.005:
            status = "over"
        elif utilisation >= 80:
            status = "approaching"
        else:
            status = "on_track"

        categories.append({
            **category,
            "budget_amount": round(budget, 2),
            "total_spending": round(spent, 2),
            "remaining_budget": round(remaining, 2),
            "utilisation_percentage": (
                round(utilisation, 2) if utilisation is not None else None
            ),
            "status": status
        })

    total_budget = sum(item["budget_amount"] for item in categories)
    total_spending = sum(item["total_spending"] for item in categories)
    return {
        "budget_month": budget_month,
        "categories": categories,
        "total_budget": round(total_budget, 2),
        "total_spending": round(total_spending, 2),
        "remaining_budget": round(total_budget - total_spending, 2)
    }


def get_monthly_spending_trend(start_date=None, end_date=None):
    """Return chronological monthly totals with month-over-month changes."""
    rows = get_monthly_trend(start_date=start_date, end_date=end_date)
    trend = []
    previous_total = None

    for row in rows:
        amount = float(row.get("total_spending") or 0)
        difference = (
            amount - previous_total if previous_total is not None else None
        )
        percentage = (
            difference / previous_total * 100
            if previous_total is not None and previous_total > 0
            else None
        )
        trend.append({
            **row,
            "total_spending": round(amount, 2),
            "difference_from_previous_month": (
                round(difference, 2) if difference is not None else None
            ),
            "percentage_change": (
                round(percentage, 2) if percentage is not None else None
            )
        })
        previous_total = amount

    return {"months": trend}


def get_monthly_spending_comparison(current_month, previous_month,
                                    category_name=None):
    """Expose the established monthly comparison under a descriptive alias."""
    return compare_monthly_spending(
        current_month=current_month,
        previous_month=previous_month,
        category_name=category_name
    )



def get_budget_overview(budget_month):
    """    Return an overall monthly budget summary and its category breakdown.

    The overview is derived exclusively from category-level allocations.
    It reuses get_category_budget_status() so category spending, remaining
    amounts, utilisation percentages and status thresholds stay consistent.

    Args:
        budget_month: Month in MM-YYYY format.

    Returns:
        A dictionary containing the overall totals and category details.
        utilisation_percentage is None when no positive budget is allocated.
    """

    # Validate the requested month before retrieving any budget information.
    month_date = datetime.strptime(budget_month, "%m-%Y")
    if month_date.strftime("%m-%Y") != budget_month:
        raise ValueError("Budget month must use MM-YYYY format.")

    # Retrieve the existing category-level calculations for this month.
    category_status = get_category_budget_status(budget_month)
    categories = category_status.get("categories", [])

    # Calculate totals from the category rows to keep this result self-consistent.
    total_budget = sum(
        float(category.get("budget_amount") or 0)
        for category in categories
    )
    total_spending = sum(
        float(category.get("total_spending") or 0)
        for category in categories
    )
    remaining_budget = total_budget - total_spending

    # An undefined percentage is more accurate than 0% when no budget exists.
    utilisation_percentage = (
        total_spending / total_budget * 100
        if total_budget > 0
        else None
    )

    return {
        "budget_month": budget_month,
        "total_budget": round(total_budget, 2),
        "total_spending": round(total_spending, 2),
        "remaining_budget": round(remaining_budget, 2),
        "utilisation_percentage": (
            round(utilisation_percentage, 2)
            if utilisation_percentage is not None
            else None
        ),
        "categories": categories,
        "budget_configured": total_budget > 0,
    }
