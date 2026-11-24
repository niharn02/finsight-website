import sqlite3
from pathlib import Path
from contextlib import closing
from datetime import datetime
import math


# Database configuration
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "expenses.db"


# Establish database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Initialise database tables
def initialise_database():

    conn = get_db_connection()

    try:
        cursor = conn.cursor()

        # Categories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            )
        """)

        # Expenses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                expense_id INTEGER PRIMARY KEY AUTOINCREMENT,
                expense_name TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                category_id INTEGER NOT NULL,
                expense_date TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            )
        """)


        # Category budgets table
        # Each category has a separate allocation for each month.
        # The overall monthly budget is calculated from these allocations.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS category_budgets (
                budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER NOT NULL,
                budget_month TEXT NOT NULL,
                budget_amount REAL NOT NULL DEFAULT 0
                    CHECK(budget_amount >= 0),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id),

                UNIQUE (category_id, budget_month)
            )
        """)

        # Period membership controls which custom categories appear in each month.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS category_periods (
                category_id INTEGER NOT NULL,
                budget_month TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories(category_id),
                UNIQUE (category_id, budget_month)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category_periods_month
            ON category_periods (budget_month)
        """)

        # Food is the only category available by default in every period.
        default_categories = [("Food",)]
        cursor.executemany("""
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
        """, default_categories)

        # Preserve existing transaction categories in the months where they were used.
        cursor.execute("""
            INSERT OR IGNORE INTO category_periods (category_id, budget_month)
            SELECT DISTINCT category_id,
                   substr(expense_date, 4, 7)
            FROM expenses
        """)

        # Create zero allocations for the Food default in every month already used.
        cursor.execute("""
            INSERT OR IGNORE INTO category_budgets
                (category_id, budget_month, budget_amount)
            SELECT c.category_id, p.budget_month, 0
            FROM categories c
            CROSS JOIN (
                SELECT DISTINCT budget_month FROM category_periods
                UNION SELECT DISTINCT substr(expense_date, 4, 7) FROM expenses
            ) p
            WHERE c.category_name IN ('Food')
        """)

        conn.commit()

        print("FinSight database initialised successfully.")

    except sqlite3.Error:
        conn.rollback()
        raise

    finally:
        conn.close()

#-------------------------------------------------
# Validate Date Formats
#-------------------------------------------------

# Validate expense date format
def validate_expense_date(expense_date):
    try:
        datetime.strptime(expense_date, "%d-%m-%Y")
        return expense_date
    except (ValueError, TypeError):
        raise ValueError("Date must be in DD-MM-YYYY format.")


# Validate that an actual expense date is not in the future
def validate_actual_expense_date(expense_date):

    # First validate the date format
    validate_expense_date(expense_date)

    # Convert the date string into a date object
    parsed_date = datetime.strptime(
        expense_date, "%d-%m-%Y"
    ).date()

    # Get today's date
    today = datetime.now().date()

    # Reject future expense dates
    if parsed_date > today:
        raise ValueError("Expense date cannot be in the future.")

    return expense_date


# Validate budget date format
def validate_budget_month(budget_month):
    """
    Validate the budget month in MM-YYYY format.
    """

    from datetime import datetime

    # Ensure the month is provided as text.
    if not isinstance(budget_month, str):
        raise ValueError("Budget month must be in MM-YYYY format.")

    try:
        # Validate both the format and actual month range.
        datetime.strptime(budget_month, "%m-%Y")

    except ValueError:
        raise ValueError(
            "Invalid budget month. Please provide a month "
            "between 01 and 12 in MM-YYYY format."
        )


# Validate category ID
def validate_category_id(category_id):
    """Check whether a category ID is valid and exists."""

    if isinstance(category_id, bool):
        raise ValueError("Category ID must be a valid integer.")

    try:
        category_id = int(category_id)
    except (ValueError, TypeError):
        raise ValueError("Category ID must be a valid integer.")

    if category_id <= 0:
        raise ValueError("Category ID must be greater than zero.")

    with closing(get_db_connection()) as conn:
        category = conn.execute("""
            SELECT category_id
            FROM categories
            WHERE category_id = ?
        """, (category_id,)).fetchone()

    if category is None:
        raise ValueError("The selected category does not exist.")

    return category_id


# Retrieve the master category list for internal reporting and chatbot use.
def get_categories():
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT category_id, category_name FROM categories
            ORDER BY category_name
        """).fetchall()
        return [dict(row) for row in rows]


# Return categories available in the selected month.
def get_categories_for_month(budget_month):
    validate_budget_month(budget_month)
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT c.category_id, c.category_name
            FROM categories c
            WHERE c.category_name IN ('Food')
               OR EXISTS (
                    SELECT 1 FROM category_periods cp
                    WHERE cp.category_id = c.category_id
                      AND cp.budget_month = ?
               )
            ORDER BY c.category_name
        """, (budget_month,)).fetchall()
        return [dict(row) for row in rows]


# Create or reuse a master category and activate it for one selected month.
def create_category(category_name, budget_month=None):
    if not isinstance(category_name, str) or not category_name.strip():
        raise ValueError("Category name must be non-empty text.")
    category_name = category_name.strip()
    if len(category_name) > 40:
        raise ValueError("Category name cannot exceed 40 characters.")
    if budget_month is not None:
        validate_budget_month(budget_month)

    with closing(get_db_connection()) as conn:
        with conn:
            row = conn.execute("""
                SELECT category_id FROM categories
                WHERE lower(category_name) = lower(?)
            """, (category_name,)).fetchone()
            if row:
                category_id = row["category_id"]
            else:
                cursor = conn.execute(
                    "INSERT INTO categories (category_name) VALUES (?)",
                    (category_name,)
                )
                category_id = cursor.lastrowid

            if budget_month:
                conn.execute("""
                    INSERT OR IGNORE INTO category_periods
                        (category_id, budget_month) VALUES (?, ?)
                """, (category_id, budget_month))
                conn.execute("""
                    INSERT OR IGNORE INTO category_budgets
                        (category_id, budget_month, budget_amount)
                    VALUES (?, ?, 0)
                """, (category_id, budget_month))
            return category_id


# Remove category availability from one month; preserve master and history.
def delete_category_for_month(category_id, budget_month):
    validate_budget_month(budget_month)
    category_id = validate_category_id(category_id)
    with closing(get_db_connection()) as conn:
        with conn:
            # Expense dates are normally stored as DD-MM-YYYY. Older or
            # imported records may use ISO YYYY-MM-DD, so normalise both
            # formats before deciding whether this category is in use.
            count = conn.execute("""
                SELECT COUNT(*)
                FROM expenses
                WHERE category_id = ?
                  AND CASE
                        WHEN length(expense_date) = 10
                             AND substr(expense_date, 3, 1) = '-'
                            THEN substr(expense_date, 4, 7)
                        WHEN length(expense_date) = 10
                             AND substr(expense_date, 5, 1) = '-'
                            THEN substr(expense_date, 6, 2) || '-' ||
                                 substr(expense_date, 1, 4)
                        ELSE NULL
                      END = ?
            """, (category_id, budget_month)).fetchone()[0]
            if count:
                raise ValueError(
                    "This category has transactions in the selected month. "
                    "Delete those transactions before removing the category."
                )
            name = conn.execute(
                "SELECT category_name FROM categories WHERE category_id = ?",
                (category_id,)
            ).fetchone()
            if name and name["category_name"] == "Food":
                raise ValueError("The default Food category cannot be removed.")
            conn.execute("""
                DELETE FROM category_periods
                WHERE category_id = ? AND budget_month = ?
            """, (category_id, budget_month))
            conn.execute("""
                DELETE FROM category_budgets
                WHERE category_id = ? AND budget_month = ?
            """, (category_id, budget_month))
    return True


#-------------------------------------------------
# Expense Functions
#-------------------------------------------------

# Add a new expense
def add_expense(expense_name, amount, category_id,
                expense_date, payment_method, notes=None):

    # Validate expense name
    if not isinstance(expense_name, str) or not expense_name.strip():
        raise ValueError("Expense name must be non-empty text.")

    # Validate amount
    if isinstance(amount, bool):
        raise ValueError("Amount must be a valid number.")

    try:
        amount = float(amount)
    except (ValueError, TypeError):
        raise ValueError("Amount must be a valid number.")

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Amount must be greater than zero.")

    # Validate payment method
    if not isinstance(payment_method, str) or not payment_method.strip():
        raise ValueError("Payment method must be non-empty text.")

    # Validate notes
    if notes is not None and not isinstance(notes, str):
        raise ValueError("Notes must be text.")

    validate_actual_expense_date(expense_date)
    category_id = validate_category_id(category_id)
    expense_month = expense_date[3:10]

    with closing(get_db_connection()) as conn:
        with conn:
            # A category used by a transaction is automatically available in that month.
            conn.execute("""
                INSERT OR IGNORE INTO category_periods (category_id, budget_month)
                VALUES (?, ?)
            """, (category_id, expense_month))
            conn.execute("""
                INSERT OR IGNORE INTO category_budgets
                    (category_id, budget_month, budget_amount)
                VALUES (?, ?, 0)
            """, (category_id, expense_month))
            cursor = conn.execute("""
                INSERT INTO expenses (
                    expense_name,
                    amount,
                    category_id,
                    expense_date,
                    payment_method,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                expense_name.strip(),
                amount,
                category_id,
                expense_date,
                payment_method.strip(),
                notes
            ))

            expense_id = cursor.lastrowid

        return expense_id

# Retrieve expenses with category names
def get_expenses():

    with closing(get_db_connection()) as conn:
        expenses = conn.execute("""
            SELECT
                e.expense_id,
                e.expense_name,
                e.amount,
                e.category_id,
                c.category_name,
                e.expense_date,
                e.payment_method,
                e.notes,
                e.created_at
            FROM expenses e
            JOIN categories c
                ON e.category_id = c.category_id
            -- Use expense_id as a tie-breaker for expenses on the same date.
            ORDER BY
                date(
                    substr(e.expense_date, 7, 4) || '-' ||
                    substr(e.expense_date, 4, 2) || '-' ||
                    substr(e.expense_date, 1, 2)
                ) DESC,
                e.expense_id DESC
        """).fetchall()

        return [dict(expense) for expense in expenses]


# Retrieve a specific expense by ID
def get_expense_by_id(expense_id):

    with closing(get_db_connection()) as conn:
        expense = conn.execute("""
            SELECT
                e.expense_id,
                e.expense_name,
                e.amount,
                e.category_id,
                c.category_name,
                e.expense_date,
                e.payment_method,
                e.notes,
                e.created_at
            FROM expenses e
            JOIN categories c
                ON e.category_id = c.category_id
            WHERE e.expense_id = ?
        """, (expense_id,)).fetchone()

        return dict(expense) if expense else None


# Update an existing expense
def update_expense(expense_id, expense_name, amount, category_id,
                   expense_date, payment_method, notes=None):

    # Validate expense name
    if not isinstance(expense_name, str) or not expense_name.strip():
        raise ValueError("Expense name must be non-empty text.")

    # Validate amount
    if isinstance(amount, bool):
        raise ValueError("Amount must be a valid number.")

    try:
        amount = float(amount)
    except (ValueError, TypeError):
        raise ValueError("Amount must be a valid number.")

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Amount must be greater than zero.")

    # Validate payment method
    if not isinstance(payment_method, str) or not payment_method.strip():
        raise ValueError("Payment method must be non-empty text.")

    # Validate notes
    if notes is not None and not isinstance(notes, str):
        raise ValueError("Notes must be text.")

    validate_actual_expense_date(expense_date)
    category_id = validate_category_id(category_id)
    expense_month = expense_date[3:10]

    with closing(get_db_connection()) as conn:
        with conn:
            # Keep the category available and budgeted in the expense's month.
            conn.execute("""
                INSERT OR IGNORE INTO category_periods (category_id, budget_month)
                VALUES (?, ?)
            """, (category_id, expense_month))
            conn.execute("""
                INSERT OR IGNORE INTO category_budgets
                    (category_id, budget_month, budget_amount)
                VALUES (?, ?, 0)
            """, (category_id, expense_month))
            cursor = conn.execute("""
                UPDATE expenses
                SET
                    expense_name = ?,
                    amount = ?,
                    category_id = ?,
                    expense_date = ?,
                    payment_method = ?,
                    notes = ?
                WHERE expense_id = ?
            """, (
                expense_name.strip(),
                amount,
                category_id,
                expense_date,
                payment_method.strip(),
                notes,
                expense_id
            ))

            return cursor.rowcount > 0



# Safely update an expense only if it has not changed since it was reviewed
def update_expense_if_unchanged(expense_id, expected_original, proposed):
    """
    Compare the current database record with the original values
    shown to the user. Update only if all original values still match.

    Returns:
        "updated"  - The expense was successfully updated.
        "not_found" - The expense no longer exists.
        "changed" - The expense was modified after the proposal was created.
    """

    # Confirm that both dictionaries contain all editable fields.
    required_fields = {
        "expense_name",
        "amount",
        "category_id",
        "expense_date",
        "payment_method",
        "notes"
    }

    if not isinstance(expected_original, dict) or not isinstance(proposed, dict):
        raise ValueError("Original and proposed expense details must be dictionaries.")

    if not required_fields.issubset(expected_original):
        raise ValueError("Original expense details are incomplete.")

    if not required_fields.issubset(proposed):
        raise ValueError("Proposed expense details are incomplete.")

    # Read the proposed values before performing any database operation.
    expense_name = proposed["expense_name"]
    amount = proposed["amount"]
    category_id = proposed["category_id"]
    expense_date = proposed["expense_date"]
    payment_method = proposed["payment_method"]
    notes = proposed["notes"]

    # Validate the expense name.
    if not isinstance(expense_name, str) or not expense_name.strip():
        raise ValueError("Expense name must be non-empty text.")

    # Validate the amount.
    if isinstance(amount, bool):
        raise ValueError("Amount must be a valid number.")

    try:
        amount = float(amount)
    except (ValueError, TypeError):
        raise ValueError("Amount must be a valid number.")

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Amount must be greater than zero.")

    # Validate the category, date, payment method and notes.
    category_id = validate_category_id(category_id)
    validate_actual_expense_date(expense_date)

    if not isinstance(payment_method, str) or not payment_method.strip():
        raise ValueError("Payment method must be non-empty text.")

    if notes is not None and not isinstance(notes, str):
        raise ValueError("Notes must be text.")

    # Convert the original numeric values for reliable comparison.
    try:
        original_amount = float(expected_original["amount"])
        original_category_id = int(expected_original["category_id"])
    except (ValueError, TypeError):
        raise ValueError("Original expense values are invalid.")

    if not math.isfinite(original_amount):
        raise ValueError("Original expense amount is invalid.")

    # Prepare the original snapshot for comparison.
    original_values = {
        "expense_name": expected_original["expense_name"],
        "amount": original_amount,
        "category_id": original_category_id,
        "expense_date": expected_original["expense_date"],
        "payment_method": expected_original["payment_method"],
        "notes": expected_original["notes"]
    }

    # Start an immediate write transaction so another writer cannot
    # change the expense between our comparison and update.
    with closing(get_db_connection()) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Retrieve the latest editable values from the database.
            current = conn.execute("""
                SELECT
                    expense_name,
                    amount,
                    category_id,
                    expense_date,
                    payment_method,
                    notes
                FROM expenses
                WHERE expense_id = ?
            """, (expense_id,)).fetchone()

            # Stop if the expense has been deleted.
            if current is None:
                conn.rollback()
                return "not_found"

            current_values = dict(current)

            # Compare every editable field with the original snapshot.
            unchanged = (
                current_values["expense_name"] == original_values["expense_name"]
                and float(current_values["amount"]) == original_values["amount"]
                and int(current_values["category_id"]) == original_values["category_id"]
                and current_values["expense_date"] == original_values["expense_date"]
                and current_values["payment_method"] == original_values["payment_method"]
                and current_values["notes"] == original_values["notes"]
            )

            # Do not overwrite a record that has changed since review.
            if not unchanged:
                conn.rollback()
                return "changed"

            # Update the record only after the original values match.
            cursor = conn.execute("""
                UPDATE expenses
                SET
                    expense_name = ?,
                    amount = ?,
                    category_id = ?,
                    expense_date = ?,
                    payment_method = ?,
                    notes = ?
                WHERE expense_id = ?
            """, (
                expense_name.strip(),
                amount,
                category_id,
                expense_date,
                payment_method.strip(),
                notes,
                expense_id
            ))

            # Commit only when the update affected a record.
            if cursor.rowcount > 0:
                conn.commit()
                return "updated"

            conn.rollback()
            return "not_found"

        except Exception:
            # Roll back any incomplete operation before propagating the error.
            conn.rollback()
            raise


# Delete an expense
def delete_expense(expense_id):

    with closing(get_db_connection()) as conn:
        with conn:
            cursor = conn.execute("""
                DELETE FROM expenses
                WHERE expense_id = ?
            """, (expense_id,))

            return cursor.rowcount > 0


# Initialise zero allocations only for categories available in this month.
def initialise_category_budgets_for_month(budget_month):
    validate_budget_month(budget_month)
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO category_budgets
                    (category_id, budget_month, budget_amount)
                SELECT category_id, ?, 0 FROM categories
                WHERE category_name IN ('Food')
                   OR category_id IN (
                        SELECT category_id FROM category_periods
                        WHERE budget_month = ?
                   )
            """, (budget_month, budget_month))


# Retrieve period-specific categories and their monthly allocations.
def get_category_budgets(budget_month):
    validate_budget_month(budget_month)
    initialise_category_budgets_for_month(budget_month)
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT c.category_id, c.category_name, cb.budget_amount
            FROM categories c
            JOIN category_budgets cb ON cb.category_id = c.category_id
            WHERE cb.budget_month = ?
              AND (
                    c.category_name IN ('Food')
                    OR EXISTS (
                        SELECT 1 FROM category_periods cp
                        WHERE cp.category_id = c.category_id
                          AND cp.budget_month = ?
                    )
              )
            ORDER BY c.category_name
        """, (budget_month, budget_month)).fetchall()
        categories = [dict(row) for row in rows]
        return {
            "budget_month": budget_month,
            "categories": categories,
            "total_budget": round(sum(float(row["budget_amount"]) for row in categories), 2)
        }


def get_category_budget(category_id, budget_month):
    """Retrieve one category allocation for a specific active month."""
    validate_budget_month(budget_month)
    category_id = validate_category_id(category_id)
    initialise_category_budgets_for_month(budget_month)

    with closing(get_db_connection()) as conn:
        row = conn.execute("""
            SELECT
                c.category_id,
                c.category_name,
                cb.budget_month,
                cb.budget_amount
            FROM categories AS c
            JOIN category_budgets AS cb
                ON cb.category_id = c.category_id
            WHERE c.category_id = ?
              AND cb.budget_month = ?
              AND (
                    c.category_name IN ('Food')
                    OR EXISTS (
                        SELECT 1 FROM category_periods AS cp
                        WHERE cp.category_id = c.category_id
                          AND cp.budget_month = ?
                    )
              )
        """, (category_id, budget_month, budget_month)).fetchone()

        return dict(row) if row else None


# Update the allocation for one category and one month only.
def update_category_budget(category_id, budget_amount, budget_month):
    validate_budget_month(budget_month)
    if isinstance(budget_amount, bool):
        raise ValueError("Budget amount must be a valid number.")
    try:
        budget_amount = float(budget_amount)
    except (ValueError, TypeError):
        raise ValueError("Budget amount must be a valid number.")
    if not math.isfinite(budget_amount) or budget_amount < 0:
        raise ValueError("Budget amount cannot be negative.")
    category_id = validate_category_id(category_id)

    with closing(get_db_connection()) as conn:
        with conn:
            available = conn.execute("""
                SELECT 1 FROM categories
                WHERE category_id = ?
                  AND (
                    category_name IN ('Food')
                    OR EXISTS (
                        SELECT 1 FROM category_periods
                        WHERE category_id = ? AND budget_month = ?
                    )
                  )
            """, (category_id, category_id, budget_month)).fetchone()
            if not available:
                raise ValueError("Add this category to the selected month first.")
            conn.execute("""
                INSERT INTO category_budgets
                    (category_id, budget_month, budget_amount)
                VALUES (?, ?, ?)
                ON CONFLICT(category_id, budget_month)
                DO UPDATE SET budget_amount = excluded.budget_amount
            """, (category_id, budget_month, budget_amount))
    return True


# Get the monthly spending and category-based budget summary.
def get_monthly_summary(budget_month):
    # Validate the requested month before querying the database.
    validate_budget_month(budget_month)

    # Convert MM-YYYY into an inclusive start and exclusive end date.
    month_date = datetime.strptime(budget_month, "%m-%Y")
    start_date = month_date.replace(day=1)
    if month_date.month == 12:
        end_date = start_date.replace(year=month_date.year + 1, month=1)
    else:
        end_date = start_date.replace(month=month_date.month + 1)

    with closing(get_db_connection()) as conn:
        # Sum expenses recorded within the selected calendar month.
        spending = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM expenses
            WHERE substr(expense_date, 7, 4) || '-' ||
                  substr(expense_date, 4, 2) || '-' ||
                  substr(expense_date, 1, 2) >= ?
              AND substr(expense_date, 7, 4) || '-' ||
                  substr(expense_date, 4, 2) || '-' ||
                  substr(expense_date, 1, 2) < ?
            """,
            (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        ).fetchone()[0]

        # The overall monthly budget is now the sum of active category allocations.
        budget_amount = conn.execute(
            """
            SELECT COALESCE(SUM(cb.budget_amount), 0)
            FROM category_budgets AS cb
            INNER JOIN category_periods AS cp
                ON cp.category_id = cb.category_id
               AND cp.budget_month = cb.budget_month
            WHERE cb.budget_month = ?
            """,
            (budget_month,)
        ).fetchone()[0]

    # Calculate the remaining amount and utilisation safely.
    remaining = budget_amount - spending
    utilisation = (spending / budget_amount) * 100 if budget_amount > 0 else None

    return {
        "budget_month": budget_month,
        "budget_amount": round(budget_amount, 2),
        "total_spending": round(spending, 2),
        "remaining_budget": round(remaining, 2),
        "utilisation_percentage": round(utilisation, 2) if utilisation is not None else None
    }


#-------------------------------------------------
# Summary
#-------------------------------------------------

# Retrieve category-wise spending with optional filters
def get_category_summary(start_date=None, end_date=None, category_id=None):

    # Validate date inputs
    if start_date:
        validate_expense_date(start_date)

    if end_date:
        validate_expense_date(end_date)

    # Validate date range
    if start_date and end_date:
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

    # Validate category ID
    if category_id is not None:
        category_id = validate_category_id(category_id)

    # Build query dynamically
    query = """
        SELECT
            c.category_id,
            c.category_name,
            SUM(e.amount) AS total_spending,
            COUNT(e.expense_id) AS transaction_count
        FROM expenses e
        JOIN categories c
            ON e.category_id = c.category_id
        WHERE 1=1
    """

    params = []

    if start_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) >= date(?)
        """

        params.append(
            datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    # Apply category filter
    if category_id is not None:
        query += " AND e.category_id = ?"
        params.append(category_id)

    if end_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) <= date(?)
        """

        params.append(
            datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    query += """
        GROUP BY c.category_id, c.category_name
        ORDER BY total_spending DESC
    """

    with closing(get_db_connection()) as conn:

        results = conn.execute(query, params).fetchall()

        total = sum(row["total_spending"] for row in results)

        summary = []

        for row in results:
            spending = row["total_spending"]

            summary.append({
                "category_id": row["category_id"],
                "category_name": row["category_name"],
                "total_spending": round(spending, 2),
                "transaction_count": row["transaction_count"],
                "percentage": round(
                    (spending / total) * 100, 2
                ) if total > 0 else 0
            })

        return summary


# Retrieve daily spending with optional filters
def get_daily_summary(
    start_date=None,
    end_date=None,
    category_id=None,
    payment_method=None
):

    # Validate dates
    if start_date:
        validate_expense_date(start_date)

    if end_date:
        validate_expense_date(end_date)

    if start_date and end_date:
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

    # Base query
    query = """
        SELECT
            e.expense_date,
            SUM(e.amount) AS total_spending,
            COUNT(e.expense_id) AS transaction_count
        FROM expenses e
        WHERE 1=1
    """

    params = []

    # Apply start date filter
    if start_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) >= date(?)
        """

        params.append(
            datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    # Apply end date filter
    if end_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) <= date(?)
        """

        params.append(
            datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    # Apply category filter
    if category_id is not None:
        category_id = validate_category_id(category_id)
        query += " AND e.category_id = ?"
        params.append(category_id)

    # Apply payment method filter
    if payment_method:
        query += " AND LOWER(e.payment_method) = LOWER(?)"
        params.append(payment_method.strip())

    # Group and sort by date
    query += """
        GROUP BY e.expense_date
        ORDER BY
            substr(e.expense_date, 7, 4),
            substr(e.expense_date, 4, 2),
            substr(e.expense_date, 1, 2)
    """

    with closing(get_db_connection()) as conn:

        results = conn.execute(query, params).fetchall()

        return [
            {
                "expense_date": row["expense_date"],
                "total_spending": round(row["total_spending"], 2),
                "transaction_count": row["transaction_count"]
            }
            for row in results
        ]

# Get Top 5 Expenses
def get_top_expenses(
    limit=5,
    start_date=None,
    end_date=None,
    category_id=None,
    payment_method=None
):
    """Return the highest expenses with optional filters."""

    # Validate the result limit
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer.")

    # Validate dates
    if start_date:
        validate_expense_date(start_date)

    if end_date:
        validate_expense_date(end_date)

    if start_date and end_date:
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

    # Build the query dynamically
    query = """
        SELECT
            e.expense_id,
            e.expense_name,
            e.amount,
            c.category_name,
            e.expense_date,
            e.payment_method,
            e.notes
        FROM expenses e
        JOIN categories c
            ON e.category_id = c.category_id
        WHERE 1=1
    """

    params = []

    if start_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) >= date(?)
        """
        params.append(datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d"))

    if end_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) <= date(?)
        """
        params.append(datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d"))

    if category_id is not None:
        category_id = validate_category_id(category_id)
        query += " AND e.category_id = ?"
        params.append(category_id)

    if payment_method:
        query += " AND LOWER(e.payment_method) = LOWER(?)"
        params.append(payment_method.strip())

    query += """
        ORDER BY e.amount DESC, e.expense_id DESC
        LIMIT ?
    """
    params.append(limit)

    with closing(get_db_connection()) as conn:
        results = conn.execute(query, params).fetchall()

        return [
            {
                "expense_id": row["expense_id"],
                "expense_name": row["expense_name"],
                "amount": row["amount"],
                "category_name": row["category_name"],
                "expense_date": row["expense_date"],
                "payment_method": row["payment_method"],
                "notes": row["notes"]
            }
            for row in results
        ]

#-------------------------------------------------
# Search Expenses
#-------------------------------------------------

def search_expenses(
    expense_name=None,
    start_date=None,
    end_date=None,
    category_id=None,
    payment_method=None
):
    """Search expenses using optional filters."""

    # Validate dates
    if start_date:
        validate_expense_date(start_date)

    if end_date:
        validate_expense_date(end_date)

    if start_date and end_date:
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

    # Validate category ID
    if category_id is not None:
        category_id = validate_category_id(category_id)

    # Build the query dynamically
    query = """
        SELECT
            e.expense_id,
            e.expense_name,
            e.amount,
            e.category_id,
            c.category_name,
            e.expense_date,
            e.payment_method,
            e.notes
        FROM expenses e
        JOIN categories c
            ON e.category_id = c.category_id
        WHERE 1=1
    """

    params = []

    # Filter by expense name
    if expense_name is not None:
        if not isinstance(expense_name, str) or not expense_name.strip():
            raise ValueError("Expense name must be non-empty text.")

        query += " AND LOWER(e.expense_name) LIKE LOWER(?)"
        params.append(f"%{expense_name.strip()}%")

    # Filter by start date
    if start_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) >= date(?)
        """
        params.append(
            datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    # Filter by end date
    if end_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) <= date(?)
        """
        params.append(
            datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    # Filter by category
    if category_id is not None:
        query += " AND e.category_id = ?"
        params.append(category_id)

    # Filter by payment method
    if payment_method:
        query += " AND LOWER(e.payment_method) = LOWER(?)"
        params.append(payment_method.strip())

    # Sort by date and expense ID
    query += """
        ORDER BY
            date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) DESC,
            e.expense_id DESC
    """

    with closing(get_db_connection()) as conn:
        results = conn.execute(query, params).fetchall()

        return [dict(row) for row in results]


# Retrieve total spending grouped by month
def get_monthly_trend(start_date=None, end_date=None):
    """Return monthly spending totals, oldest month first."""

    if start_date:
        validate_expense_date(start_date)

    if end_date:
        validate_expense_date(end_date)

    if start_date and end_date:
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

    # Dates are DD-MM-YYYY text, so characters 4-10 are 'MM-YYYY'
    query = """
        SELECT
            substr(e.expense_date, 4, 7) AS month,
            SUM(e.amount) AS total_spending,
            COUNT(e.expense_id) AS transaction_count
        FROM expenses e
        WHERE 1=1
    """

    params = []

    if start_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) >= date(?)
        """
        params.append(
            datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    if end_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) <= date(?)
        """
        params.append(
            datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    # Sort chronologically: year first, then month
    query += """
        GROUP BY month
        ORDER BY
            substr(e.expense_date, 7, 4),
            substr(e.expense_date, 4, 2)
    """

    with closing(get_db_connection()) as conn:
        results = conn.execute(query, params).fetchall()

        return [
            {
                "month": row["month"],
                "total_spending": round(row["total_spending"], 2),
                "transaction_count": row["transaction_count"]
            }
            for row in results
        ]

#-------------------------------------------------
# Payment Summary
#-------------------------------------------------

# Get Payment Summary
def get_payment_method_summary(
    start_date=None,
    end_date=None,
    category_id=None
):
    """Return spending breakdown by payment method."""

    # Validate dates
    if start_date:
        validate_expense_date(start_date)

    if end_date:
        validate_expense_date(end_date)

    if start_date and end_date:
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

    # Build query dynamically
    query = """
        SELECT
            e.payment_method,
            SUM(e.amount) AS total_spending,
            COUNT(e.expense_id) AS transaction_count
        FROM expenses e
        WHERE 1=1
    """

    params = []

    if start_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) >= date(?)
        """
        params.append(
            datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    if end_date:
        query += """
            AND date(
                substr(e.expense_date, 7, 4) || '-' ||
                substr(e.expense_date, 4, 2) || '-' ||
                substr(e.expense_date, 1, 2)
            ) <= date(?)
        """
        params.append(
            datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    if category_id is not None:
        category_id = validate_category_id(category_id)
        query += " AND e.category_id = ?"
        params.append(category_id)

    query += """
        GROUP BY e.payment_method
        ORDER BY total_spending DESC
    """

    with closing(get_db_connection()) as conn:
        results = conn.execute(query, params).fetchall()

        total_spending = sum(row["total_spending"] for row in results)

        return [
            {
                "payment_method": row["payment_method"],
                "total_spending": round(row["total_spending"], 2),
                "transaction_count": row["transaction_count"],
                "percentage": round(
                    (row["total_spending"] / total_spending) * 100, 2
                ) if total_spending > 0 else 0
            }
            for row in results
        ]

#-------------------------------------------------
# Dashboard Summary
#-------------------------------------------------

# Get Dashboard Summary
def get_dashboard_summary(start_date=None, end_date=None):
    """Return consolidated expense metrics for the dashboard."""

    # Validate dates
    if start_date:
        validate_expense_date(start_date)

    if end_date:
        validate_expense_date(end_date)

    if start_date and end_date:
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

    # Build date filters
    date_condition = ""
    params = []

    if start_date:
        date_condition += """
            AND date(
                substr(expense_date, 7, 4) || '-' ||
                substr(expense_date, 4, 2) || '-' ||
                substr(expense_date, 1, 2)
            ) >= date(?)
        """
        params.append(
            datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    if end_date:
        date_condition += """
            AND date(
                substr(expense_date, 7, 4) || '-' ||
                substr(expense_date, 4, 2) || '-' ||
                substr(expense_date, 1, 2)
            ) <= date(?)
        """
        params.append(
            datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        )

    with closing(get_db_connection()) as conn:

        # Overall expense metrics
        summary = conn.execute(
            f"""
            SELECT
                COUNT(*) AS transaction_count,
                COALESCE(SUM(amount), 0) AS total_expenses,
                COALESCE(AVG(amount), 0) AS average_expense
            FROM expenses
            WHERE 1=1 {date_condition}
            """,
            params
        ).fetchone()

        # Highest spending category
        highest_category = conn.execute(
            f"""
            SELECT
                c.category_name,
                SUM(e.amount) AS total_spending
            FROM expenses e
            JOIN categories c
                ON e.category_id = c.category_id
            WHERE 1=1
            {date_condition.replace("expense_date", "e.expense_date")}
            GROUP BY c.category_id, c.category_name
            ORDER BY total_spending DESC
            LIMIT 1
            """,
            params
        ).fetchone()

        # Largest individual expense
        largest_expense = conn.execute(
            f"""
            SELECT
                e.expense_id,
                e.expense_name,
                e.amount,
                c.category_name,
                e.expense_date
            FROM expenses e
            JOIN categories c
                ON e.category_id = c.category_id
            WHERE 1=1
            {date_condition.replace("expense_date", "e.expense_date")}
            ORDER BY e.amount DESC, e.expense_id DESC
            LIMIT 1
            """,
            params
        ).fetchone()

        return {
            "total_expenses": round(summary["total_expenses"], 2),
            "transaction_count": summary["transaction_count"],
            "average_expense": round(summary["average_expense"], 2),
            "highest_spending_category": (
                {
                    "category_name": highest_category["category_name"],
                    "total_spending": round(
                        highest_category["total_spending"], 2
                    )
                }
                if highest_category else None
            ),
            "largest_expense": (
                dict(largest_expense)
                if largest_expense else None
            )
        }


# Run initialisation when executed directly
if __name__ == "__main__":
    initialise_database()
