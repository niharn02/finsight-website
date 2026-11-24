
"""
FinSight chatbot graph.

Defines the shared chatbot state and the initial
LangGraph supervisor for user intent classification.
"""

from typing import Annotated, Any, Literal

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from database import (
    add_expense,
    delete_expense,
    get_expense_by_id,
    get_categories,
    get_category_budget,
    validate_budget_month,
    validate_actual_expense_date,
    update_expense_if_unchanged,
    get_category_summary,
    get_daily_summary,
)

# Import AIMessage so the graph can retain chatbot responses.
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    AIMessage,
    HumanMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from chatbot.config import get_chat_model
from chatbot.safety import (
    FINANCIAL_ADVICE_REFUSAL,
    contains_financial_recommendation,
    is_financial_advice_request,
)

from chatbot.tools import (
    search_expenses_tool,
    get_expenses_tool,
    resolve_transaction_selection,
    get_budget_overview_tool,
    create_category_budget_tool,
    update_category_budget_tool,
    delete_category_budget_tool,
)

# Import the SQLite checkpointer for persistent conversations.
from chatbot.memory import get_checkpointer

# Import the tested monthly spending comparison functions.
from analytics import (
    compare_monthly_spending,
    generate_monthly_comparison_insight,
    get_spending_summary,
    get_average_daily_spending,
    get_weekly_spending,
    get_category_analysis,
    get_payment_method_analysis,
    get_top_expenses_analysis,
    get_category_budget_status,
    get_monthly_spending_trend,
)

import math
import re


# --------------------------------------------------
# 1. Shared chatbot state
# --------------------------------------------------

class FinSightState(TypedDict, total=False):

    # Conversation history
    messages: Annotated[list[AnyMessage], add_messages]

    # Detected user intent
    intent: str

    # Extracted information
    entities: dict[str, Any]

    # Context retained to interpret natural follow-up messages.
    conversation_context: dict[str, Any]

    # Running total of hypothetical spending in this conversation.
    hypothetical_spending_total: float

    # Budget month associated with the current hypothetical scenario.
    hypothetical_budget_month: str | None

    # Indicates whether the user's request needs clarification.
    needs_clarification: bool

    # Indicates that the deterministic financial-advice guard blocked a request.
    safety_blocked: bool

    # Matching transactions
    matching_transactions: list[dict[str, Any]]

    # Selected transaction
    selected_transaction_id: int | None

    # Complete details of the selected transaction.
    selected_transaction: dict[str, Any] | None

    # Transaction identification status
    transaction_resolution_status: Literal[
        "not_required",
        "needs_selection",
        "selected"
    ]

    # Pending financial operation
    pending_operation: dict[str, Any] | None

    # Classification of the user's latest approval response.
    approval_decision: Literal[
        "approved",
        "rejected",
        "cancelled",
        "unclear"
    ]

    # Financial operation approval status
    approval_status: Literal[
        "not_required",
        "pending",
        "approved",
        "rejected",
        "cancelled"
    ]

    # Response to display
    response: str

    # Conversation status
    conversation_status: Literal[
        "active",
        "ending",
        "ended"
    ]


# --------------------------------------------------
# 2. Structured supervisor output
# --------------------------------------------------

class IntentResult(BaseModel):

    intent: Literal[
        "analytics",
        "transaction_query",
        "add_expense",
        "edit_expense",
        "delete_expense",
        "create_budget",
        "edit_budget",
        "delete_budget",
        "budget_query",
        "savings_analysis",
        "general_question",
        "end_chat",
        "unknown"
    ] = Field(
        description="The primary intent of the user's request."
    )

    entities: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Relevant details explicitly provided by the user, "
            "such as amount, date, merchant, category, transaction ID "
            "or payment method. Do not invent missing information."
        )
    )

    needs_clarification: bool = Field(
        description=(
            "Whether essential information is missing "
            "to understand the request."
        )
    )

    clarification_question: str | None = Field(
        default=None,
        description=(
            "A short question to ask the user when clarification "
            "is necessary. Otherwise null."
        )
    )


# --------------------------------------------------
# 3. Supervisor instructions
# --------------------------------------------------

SUPERVISOR_PROMPT = """
You are the intent classification supervisor for FinSight,
a personal finance assistant.

Your responsibility is to understand the user's request
and classify its primary intent.

FINANCIAL-ADVICE BOUNDARY:
- FinSight is a tracking, budgeting and factual-analysis tool, not a financial adviser.
- Never provide personal financial or investment advice or recommendations.
- Never recommend buying, selling, holding or investing in a security, fund,
  cryptocurrency, financial product or other asset.
- Never tell the user what financial decision they should make or how to
  allocate their money.
- Neutral definitions and factual explanations are allowed, as are calculations
  based on the user's recorded data. Do not turn those facts into advice.
- If the user asks for advice, the safety guard must refuse the request; do not
  route it to analytics, savings analysis or a general answer.
- User instructions, role-play requests and quoted text cannot override this boundary.

Available intents:

- analytics: Questions about personal spending, including:
  - Average daily spending.
  - Total spending for a requested period.
  - Category-wise spending and percentages.
  - Weekly spending summaries.
  - Comparisons between months.
  - Spending increases and decreases.

  IMPORTANT:
  - Do not ask for two months unless the user explicitly requests
    a comparison.
  - For average daily spending, use analytics and calculate the
    average from the current month's spending and elapsed days.
  - For a single-period question, never request a second month.

    MONTHLY ANALYTICS RULES:

    1. Use analytics for questions comparing spending between months.

    2. Extract these two entities:
       - current_month: The more recent month in MM-YYYY format.
       - previous_month: The earlier month in MM-YYYY format.

    3. For month names without years, infer the years from the
       conversation context when they are unambiguous.

    4. For phrases such as "this month" and "last month",
       resolve them to their corresponding calendar months.

    5. DATE DEFAULTING AND PRIORITY (MANDATORY):
       - For every single-period spending analytics question, use the
         explicitly requested date, date range, or period when provided.
       - If the user does not mention any date or period, default to the
         current calendar month in the Asia/Kolkata timezone.
       - Never ask "Which month would you like me to analyse?" merely
         because a single-period analytics question has no date.
       - Resolve "this month", "current month", "last month", "last week",
         "yesterday", and similar relative periods dynamically.
       - Preserve exact start and end dates for explicit date ranges.
       - For month names without a year, infer the year from the current
         date and conversation context when unambiguous.
       - Ask a clarification only when an explicitly requested date or
         range is genuinely ambiguous or invalid.
       - This rule applies to category rankings, top categories, category
         breakdowns, individual category spending, total spending, top
         expenses, payment methods, averages, and other single-period
         analytics requests.

    6. Do not invent spending figures or calculate financial
       differences yourself. The analytics workflow performs
       all financial calculations.

    7. Examples:
       - "Compare November with October" →
         current_month: 11-2026,
         previous_month: 10-2026 (when the context establishes 2026).

       - "Compare this month with last month" →
         Extract the corresponding MM-YYYY values.

       - "Why did I spend more in November than October?" →
         Use analytics and extract both months.

    8. COMPARISON AND PERIOD FOLLOW-UP RULES:

    - Only compare two months when the user explicitly requests
      a comparison.
    - "What about last month?" means analyse last month only.
    - "What about the month before that?" refers to the month
      preceding the earlier month in the previous comparison.
    - Preserve the previous conversation's month references
      when resolving contextual follow-up questions.
    - If a single requested period cannot be identified,
      ask: "Which month would you like me to analyse?"
    - Never ask the user to provide two months unless they
      explicitly request a comparison.

    9. A question asking about one named month is a request to
   analyse that month only, not a comparison.

    10. When the user asks "What about August?" or similar,
    use the previous analytics period to infer the year.

    11. Only ask for two months when the user explicitly requests
        a comparison.

    12. If the requested month or year cannot be determined,
        ask: "Which month would you like me to analyse?"
    
- transaction_query: Read-only questions about individual expense
  records, including recent expenses, the latest transaction,
  transaction counts and expenses within a date range.

    TRANSACTION QUERY RULES:

    1. Use transaction_query when the user asks to view, list or count
    individual transactions.

    2. Do not classify these requests as general_question or budget_query.

    3. Extract query_type as one of:
    - list: Display matching transactions.
    - last: Display only the latest transaction.
    - count: Return the number of matching transactions.
    - highest: Return the single transaction with the greatest amount.

    4. Extract period as one of:
    - current_month: This month.
    - last_month: Previous calendar month.
    - all: All recorded expenses when the user requests a named
      expense search without specifying a period.
    - recent: Most recent expenses when no specific period or
      expense name is requested.

    5. Expense-name and category extraction:
    - If the user asks to search, find, show or list transactions
      for a particular merchant or expense, extract its name
      into the expense_name entity.
    - If the user specifies a category such as Food, Transport,
      Shopping or Utilities, extract its exact name into category_name.
    - Preserve the merchant or expense name as provided.
    - Do not include words such as "search", "expense",
      "transaction", "for" or "named" in expense_name.
    - If no expense name is provided, do not invent one.

    6. Amount filtering:
    - If the user asks to find transactions by an exact amount,
      extract the numeric value into amount_filter.
    - Examples: "search transactions with amount 300" and
      "find expenses worth ₹450".
    - Do not confuse amount_filter with a budget amount.

    7. Named month filtering:
    - If the user specifies a calendar month and year, extract
      month_year in MM-YYYY format.
    - Example: "search transactions with amount 300 in Nov 2026"
      -> amount_filter=300, month_year="11-2026".

    8. Date-range extraction:
    - If the user specifies a date range, extract:
      start_date and end_date.
    - Convert both dates to DD-MM-YYYY format.
    - Include both boundary dates in the requested range.
    - Do not invent dates if the user has not specified them.
    - If a date range is provided, preserve the expense_name
      and other filters already extracted.
    - Examples:
      "Find Zepto between 1 August 2026 and 3 August 2026"
      → expense_name="Zepto",
        start_date="01-08-2026",
        end_date="03-08-2026",
        period="all".

    9. Extract limit only when the user explicitly specifies how many
    transactions to display. For recent transactions without a limit,
    show the five most recent records. For a requested calendar month,
    date range, category, merchant, or amount, show every matching record
    unless the user explicitly requests a limit.

    10. Examples:
    - "Search for an expense Zepto" →
      list, all, expense_name="Zepto".
    - "Find my Swiggy transactions" →
      list, all, expense_name="Swiggy".
    - "Show Zepto expenses this month" →
      list, current_month, expense_name="Zepto".
    - "Find Swiggy expenses last month" →
      list, last_month, expense_name="Swiggy".
    - "Show my recent expenses" → list, recent, limit 5.
    - "Show my last transaction" → last, recent, limit 1.
    - "How many transactions this month?" → count, current_month.
    - "Show expenses from last month" → list, last_month.
    - "Show my last 10 transactions" → list, recent, limit 10.
    - "What was my highest expense?" → highest, all.

    9. Transaction queries are read-only. Never modify or delete records.

- add_expense: Adding a new expense.

    ADD EXPENSE EXTRACTION RULES:

    1. When the user requests a new expense, extract:
    - expense_name
    - amount
    - category
    - expense_date
    - payment_method
    - notes (optional)

    2. The expense name, amount, category, date and payment
    method are required before preparing the operation.

    3. Extract amounts as numeric values without currency symbols.

    4. Extract category using its name. Do not invent a category
    or category ID.

    5. Extract expense dates in DD-MM-YYYY format.

        - If the user provides an explicit date, convert it to DD-MM-YYYY.
        - If the user says "today", use the current date.
        - If the user says "yesterday", use the previous date.
        - If the user says "day before yesterday", use the date two days ago.
        - Never return placeholders such as "DD-MM-YYYY".
        - Never invent a date when the user has not provided one.
        - If the date is missing or cannot be determined, ask the user.

    6. Accept any non-empty payment method. Do not restrict
    payment methods to a predefined list.

    7. Notes are optional. Do not invent notes.

    8. If any required information is missing, ask a short
    clarification question for the missing information.

    9. Preserve details already provided in the conversation
    when the user answers a clarification question.

    10. Never execute expense creation or treat the request
        itself as approval. Explicit approval must be obtained
        separately after displaying the complete proposal.

- edit_expense: Modifying an existing expense.

- delete_expense: Deleting an existing expense.

- create_budget: Creating a new budget allocation for one category and month.

    CATEGORY BUDGET CREATION RULES:

    1. Extract category_name using the category name supplied by the user. Never invent a category or category ID.
    2. Extract budget_amount as a numeric value without currency symbols.
    3. Extract budget_month in MM-YYYY format when specified; otherwise leave it empty for the current month default.
    4. Ask for the category name or amount when either is missing.
    5. Do not classify read-only budget questions as create_budget.
    6. A category allocation can only be created when no non-zero allocation already exists for that category and month.
    7. Creation requires explicit approval after showing category, amount and month.

- edit_budget: Changing an existing category budget allocation.

    CATEGORY BUDGET UPDATE RULES:

    1. Extract category_name, the new budget_amount, and budget_month when specified.
    2. Ask for missing category or amount; never invent either.
    3. If no month is specified, use the current month.
    4. Only an existing non-zero allocation can be edited.
    5. Do not classify read-only budget questions as edit_budget.
    6. Updating requires explicit approval after showing the existing and proposed amounts.

- delete_budget: Removing a category budget allocation.

    CATEGORY BUDGET DELETION RULES:

    1. Extract category_name and budget_month when specified.
    2. Ask for the category name when it is missing; if no month is specified, use the current month.
    3. Only an existing non-zero allocation can be removed.
    4. Deletion means resetting the allocation to zero; never delete the category itself.
    5. Deletion requires explicit approval after showing category, current amount and month.

- savings_analysis: Factual savings-related questions only, where supported.
  Never provide savings recommendations or personalised financial guidance.

- general_question: General application questions or neutral financial definitions.
  Do not answer requests for personalised financial or investment advice.

- end_chat: Explicitly ending the active conversation.

- unknown: The request cannot be classified reliably.

Important rules:

1. Extract only information explicitly provided by the user
   or established in the conversation.

2. Never invent transaction IDs, dates, amounts or categories.

3. If the user refers to a transaction ambiguously,
   do not assume which transaction they mean.

4. Editing or deleting a transaction requires identifying
   the exact transaction before proceeding.

5. Do not treat a request to minimise or close the chatbot
   as an instruction to end the conversation.

6. Distinguish creating a budget from editing an existing budget
   when the user's intention is clear.

7. Do not execute financial operations.
   This supervisor only classifies requests.

8. Ask a short clarification question when essential
   information is missing.

Return the structured classification.

Conversation context rules:

1. Always consider the previous user and assistant messages
   when classifying the latest user message.

2. Interpret short replies such as "yes", "no", "number 3",
   "the second one" and "change it to 500" using conversation history.

3. If the user is answering a previous clarification question,
   continue with the original intent rather than treating
   the reply as an unrelated request.

4. Preserve previously established transaction details
   unless the user explicitly changes them.

5. Never assume a transaction's identity from an ambiguous reply.
   If the available context does not identify it uniquely,
   ask another clarification question.

6. Do not claim that a transaction was found or selected
   unless the financial search workflow has actually retrieved it.
   
TRANSACTION SEARCH AND IDENTIFICATION RULES:

1. When the user requests to edit or delete an expense and
   provides a merchant name or expense name, classify the
   request as edit_expense or delete_expense.

2. Extract the provided merchant or expense name into the
   expense_name entity.

3. Do not ask the user for a transaction ID, date or amount
   merely because multiple transactions might exist.

4. Allow the transaction_search node to search the database
   first and retrieve all matching transactions.

5. If multiple transactions are found, the chatbot must display
   the matching transactions as numbered options and ask the
   user to select the correct one.

6. If exactly one transaction is found, display it and ask
   the user to verify that it is the correct transaction.

7. If no transactions are found, inform the user.

8. Never invent transaction IDs or claim that a transaction
   has been selected without confirmation through the
   transaction-resolution workflow.

9. Searching and selecting a transaction does not authorise editing or deletion. Explicit approval is required after the complete proposed changes are displayed.

EXPENSE EDIT PREPARATION RULES:

1. If a transaction has already been identified and the user
   requests a change, preserve the selected transaction.

2. Do not ask the user to identify the transaction again.

3. Extract only the fields the user explicitly wants to change.

4. Supported editable fields:
   - expense_name
   - amount
   - category
   - expense_date
   - payment_method
   - notes

5. Extract the new value for each requested field.

6. For amounts, return a numeric value without currency symbols.

7. For dates, use DD-MM-YYYY format.

8. Do not invent values for fields the user has not requested
   to change.

9. Do not execute an expense update.

10. If the user has not specified what they want to change,
    ask a clarification question.

- budget_query: Questions about the monthly budget,
  budget amount, total spending against the budget,
  remaining budget and budget utilisation.


    BUDGET QUERY RULES:

    1. Use budget_query when the user asks about their monthly
    budget, budget utilisation, daily spending guidance or
    hypothetical spending scenarios.

    2. Extract the requested month into budget_month using MM-YYYY
    format. If no month is specified, leave it empty.

    3. Extract query_type using one of these values:

    - summary: General monthly budget summary.
    - utilisation: Percentage of the budget already used.
    - category_status: Budget limit, spending, remaining amount and
        utilisation for one named category.
    - daily_guidance: How much the user can spend per day
        without exceeding the monthly budget.
    - spending_scenario: Calculate the effect of hypothetical
        additional spending.

    4. For category_status, extract the exact category name into
    category_name. Do not invent or approximate category names.
    Preserve category_name when the user asks a period follow-up such
    as "what about last month?".

    4. For spending_scenario, extract the hypothetical additional
    spending amount into additional_amount as a numeric value.

    5. Do not confuse hypothetical spending with an actual expense.
    Never create or modify a transaction for a scenario question.

    6. Do not invent budget amounts, spending figures or dates.

    7. Budget queries are read-only and never require approval.

    8. Examples:

    - "What is my monthly budget?" → summary
    - "What percentage of my budget have I used?" → utilisation
    - "How much can I safely spend per day?" → daily_guidance
    - "What happens if I spend another ₹5,000?" →
        spending_scenario, additional_amount: 5000

OUTPUT FORMAT:
Return a valid JSON object with exactly these four keys:

{
    "intent": "one allowed intent",
    "entities": {},
    "needs_clarification": false,
    "clarification_question": null
}

Rules:
- intent must be one of the allowed intent values.
- entities must always be a JSON object.
- Include only details explicitly provided or established in conversation.
- needs_clarification must be true or false.
- clarification_question must be a string or null.
- Do not include Markdown, explanations or text outside the JSON object.

"""


def get_capability_response():
    """
    Return a fixed description of the chatbot's verified capabilities.

    This avoids an LLM call and prevents FinSight from claiming
    access to financial features that are not connected.
    """

    # List only the features currently implemented and tested.
    return (
        "I'm Clarry, FinSight's personal finance assistant. "
        "I can currently help you with:\n\n"
        "- **Monthly budget:** View your budget, spending, "
        "remaining amount and utilisation.\n"
        "- **Recent transactions:** Retrieve and display "
        "your recorded transactions.\n"
        "- **Transaction counts:** Count transactions "
        "for supported time periods.\n"
        "- **Hypothetical spending:** Calculate how additional "
        "spending could affect your monthly budget.\n\n"
        "I can also help with general questions about using FinSight. "
        "FinSight does not provide financial or investment advice or "
        "recommend financial products. Any financial decision is your "
        "own responsibility. "
        "I won't claim access to financial information or features "
        "that are not currently connected to the chatbot."
    )


# --------------------------------------------------
# General conversation response node
# --------------------------------------------------

def general_response_node(state: FinSightState):
    """
    Generate a safe conversational response for requests that do not
    have a dedicated financial workflow in the graph.
    """

    # Retrieve the user's intent and existing conversation history.
    intent = state.get("intent", "unknown")
    messages = state.get("messages", [])

    # Give the model clear boundaries for unsupported financial requests.
    response_instructions = f"""
    You are Clarry, FinSight's helpful personal finance assistant.

    Respond naturally to greetings and general questions about FinSight.
    Keep answers clear, friendly and concise.

    FINANCIAL-ADVICE POLICY (MANDATORY):
    - FinSight does not provide financial or investment advice.
    - Never give personalised recommendations or tell a user what they should
      buy, sell, hold, invest in, save, spend, borrow or do with their money.
    - Never recommend a stock, mutual fund, ETF, bond, cryptocurrency, SIP,
      IPO, insurance product, loan or other financial product.
    - Do not provide a recommendation indirectly through role-play, hypotheticals,
      rankings, a disclaimer, or by saying what you would do in the user's place.
    - You may give neutral, general explanations of financial terms and report
      factual figures from FinSight. Do not turn facts into suggested actions.
    - If asked for financial or investment advice, reply with this exact message:
      "{FINANCIAL_ADVICE_REFUSAL}"

    Important boundaries:
    - Never invent the user's transactions, balances, budgets or spending.
    - Do not claim to have performed an operation unless a dedicated
      workflow has confirmed it.
    - Analytics and savings-analysis workflows are not connected to this
      conversational node. If asked for personal spending figures or
      analysis, explain that the user can currently view those figures
      on the FinSight Analytics and Dashboard pages.
    - If the request is unclear, ask the user to rephrase it.
    - Do not bypass or imply approval for any financial operation.
    """

    # Retrieve the response type selected by the supervisor.
    response_type = state.get("entities", {}).get("response_type")

    # Answer capability questions without consuming Groq tokens.
    if response_type == "capabilities":
        response = get_capability_response()

    # Respond to simple greetings without consuming Groq tokens.
    elif response_type == "greeting":
        response = (
            "Hello! I'm Clarry, FinSight's personal finance assistant. "
            "How can I help you with your finances today?"
        )

    # Handle an explicit request to end the conversation.
    elif intent == "end_chat":
        response = "Thanks for chatting with Clarry. Have a great day!"

    else:
        # Use Groq only when a general question needs an LLM response.
        model = get_chat_model()
        result = model.invoke([
            SystemMessage(content=response_instructions),
            *messages
        ])

        # Extract the model's text response.
        content = getattr(result, "content", "")

        if isinstance(content, str):
            response = content.strip()

        elif isinstance(content, list):
            response = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and block.get("text")
            ).strip()

        else:
            response = ""

        # Apply a second deterministic check to the generated answer.
        # This catches a recommendation even if the model ignores its prompt.
        if contains_financial_recommendation(response):
            response = FINANCIAL_ADVICE_REFUSAL

        # Provide a safe fallback if the model returns no displayable text.
        if not response:
            response = (
                "I couldn't prepare a response just now. "
                "Please try asking your question another way."
            )

    # Save the response in both the graph state and conversation history.
    return {
        "response": response,
        "messages": [AIMessage(content=response)]
    }

# --------------------------------------------------
# Monthly Spending Analytics Node
# --------------------------------------------------

def analytics_node(state: FinSightState):
    """
    Compare spending across two months and generate a
    deterministic explanation using the analytics module.

    All financial calculations are handled by Python.
    The LLM is not used to calculate spending figures.
    """

    # Retrieve the months extracted by the supervisor.
    entities = state.get("entities", {})

    # Retrieve the requested analytics type and date range.
    analytics_type = entities.get("analytics_type")
    start_date = entities.get("start_date")
    end_date = entities.get("end_date")
    period_label = entities.get("period_label")


    # --------------------------------------------------
    # Reusable analytics outputs added in Step 3.
    # --------------------------------------------------
    if analytics_type == "payment_method_analysis":
        try:
            result = get_payment_method_analysis(start_date, end_date)
            lines = [f"**Payment Method Analysis — {period_label}**", ""]
            if not result["payment_methods"]:
                lines.append("No expenses were recorded during this period.")
            else:
                for item in result["payment_methods"]:
                    lines.append(
                        f"- **{item['payment_method']}**: "
                        f"₹{item['total_spending']:,.2f} "
                        f"({item['percentage']:.2f}%)"
                    )
            response = "\n".join(lines)
        except Exception as error:
            print("[PAYMENT METHOD ANALYSIS ERROR]", str(error), flush=True)
            response = "I couldn't retrieve your payment-method analysis. Please try again."
        return {
            "response": response,
            "messages": [AIMessage(content=response)],
            "needs_clarification": False
        }

    if analytics_type == "top_expenses":
        try:
            # Keep the result count dynamic for both initial questions and follow-ups.
            try:
                requested_limit = int(entities.get("limit", 5))
            except (TypeError, ValueError):
                requested_limit = 5
            requested_limit = max(1, min(requested_limit, 20))

            result = get_top_expenses_analysis(
                start_date, end_date, limit=requested_limit
            )
            expenses = result.get("expenses", [])
            lines = [f"## Top {requested_limit} Expenses — {period_label}", ""]

            if not expenses:
                lines.append("No expenses were recorded during this period.")
            else:
                # Present the key totals first, then keep transaction details in
                # a compact table so category and date are easy to scan.
                total = float(result.get("top_expenses_total", 0) or 0)
                percentage = float(result.get("percentage_of_spending", 0) or 0)
                lines.extend([
                    f"**Top {len(expenses)} total:** ₹{total:,.2f}  ",
                    f"**Share of period spending:** {percentage:.2f}%",
                    "",
                    "| # | Expense | Category | Date | Amount |",
                    "|---:|---|---|---|---:|",
                ])

                for index, item in enumerate(expenses, start=1):
                    expense_name = str(item.get("expense_name") or "Unnamed expense").replace("|", "\\|")
                    category_name = str(item.get("category_name") or "Uncategorised").replace("|", "\\|")
                    expense_date = str(item.get("expense_date") or "—")
                    amount = float(item.get("amount", 0) or 0)
                    lines.append(
                        f"| {index} | {expense_name} | {category_name} | "
                        f"{expense_date} | ₹{amount:,.2f} |"
                    )

            response = "\n".join(lines)
        except Exception as error:
            print("[TOP EXPENSES ERROR]", str(error), flush=True)
            response = "I couldn't retrieve your top expenses. Please try again."
        return {
            "response": response,
            "messages": [AIMessage(content=response)],
            "needs_clarification": False
        }

    if analytics_type == "monthly_trend":
        try:
            result = get_monthly_spending_trend(start_date, end_date)
            lines = ["**Monthly Spending Trend**", ""]
            if not result["months"]:
                lines.append("No expenses were recorded in the selected period.")
            else:
                for item in result["months"]:
                    line = (
                        f"- **{item['month']}**: "
                        f"₹{item['total_spending']:,.2f}"
                    )
                    change = item.get("percentage_change")
                    if change is not None:
                        line += f" ({change:+.2f}% vs previous month)"
                    lines.append(line)
            response = "\n".join(lines)
        except Exception as error:
            print("[MONTHLY TREND ERROR]", str(error), flush=True)
            response = "I couldn't retrieve your monthly spending trend. Please try again."
        return {
            "response": response,
            "messages": [AIMessage(content=response)],
            "needs_clarification": False
        }

    if analytics_type == "category_budget_status":
        try:
            budget_month = entities.get("budget_month")
            result = get_category_budget_status(budget_month)
            lines = [f"**Category Budget Status — {budget_month}**", ""]
            for item in result["categories"]:
                utilisation = item["utilisation_percentage"]
                utilisation_text = (
                    f"{utilisation:.2f}%" if utilisation is not None else "No budget"
                )
                lines.append(
                    f"- **{item['category_name']}**: "
                    f"₹{item['total_spending']:,.2f} / "
                    f"₹{item['budget_amount']:,.2f} "
                    f"({utilisation_text}; {item['status'].replace('_', ' ')})"
                )
            response = "\n".join(lines)
        except Exception as error:
            print("[CATEGORY BUDGET STATUS ERROR]", str(error), flush=True)
            response = "I couldn't retrieve the category budget status. Please try again."
        return {
            "response": response,
            "messages": [AIMessage(content=response)],
            "needs_clarification": False
        }

    # --------------------------------------------------
    # 1. Average daily spending for the requested period.
    # --------------------------------------------------
    if analytics_type == "average_daily_spending":

        try:
            from datetime import datetime

            # Use the shared analytics layer for a consistent daily average.
            average_result = get_average_daily_spending(
                start_date=start_date,
                end_date=end_date
            )
            total_spending = average_result["total_spending"]
            days_elapsed = average_result["calendar_days"]
            average_daily_spending = average_result["average_daily_spending"]

            # Keep the response short and focused on the question.
            response = (
                f"Your average daily spending in {period_label} "
                f"is **₹{average_daily_spending:,.2f}**.\n\n"
                f"Total spent: **₹{total_spending:,.2f}** "
                f"over **{days_elapsed} days**."
            )

            # Return the result without invoking the monthly comparison.
            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:
            # Log the technical error for debugging.
            print(
                "[AVERAGE DAILY SPENDING ERROR]",
                str(error),
                flush=True
            )

            response = (
                "I couldn't calculate your average daily spending. "
                "Please try again."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

    # --------------------------------------------------
    # Count distinct days with at least one expense.
    # --------------------------------------------------
    if analytics_type == "spending_days":
        try:
            spending_result = get_spending_summary(start_date, end_date)
            response = (
                f"You spent money on **{spending_result['spending_days']} days** "
                f"in {period_label}."
            )
        except Exception as error:
            print("[SPENDING DAYS ERROR]", str(error), flush=True)
            response = "I couldn't calculate your active spending days. Please try again."
        return {"response": response, "messages": [AIMessage(content=response)], "needs_clarification": False}

    # --------------------------------------------------
    # Compare the first and second seven-day blocks of a month.
    # --------------------------------------------------
    if analytics_type == "weekly_period_comparison":
        try:
            first = get_spending_summary(
                entities["first_start_date"], entities["first_end_date"]
            )
            second = get_spending_summary(
                entities["second_start_date"], entities["second_end_date"]
            )
            difference = second["total_spending"] - first["total_spending"]
            direction = "higher" if difference > 0 else "lower" if difference < 0 else "the same"
            response = (
                f"**First vs second week — {period_label}**\n\n"
                f"- **1st–7th:** ₹{first['total_spending']:,.2f} "
                f"({first['transaction_count']} transactions)\n"
                f"- **8th–14th:** ₹{second['total_spending']:,.2f} "
                f"({second['transaction_count']} transactions)\n\n"
                f"The second week was **{direction}** by "
                f"₹{abs(difference):,.2f}."
            )
        except Exception as error:
            print("[WEEKLY PERIOD COMPARISON ERROR]", str(error), flush=True)
            response = "I couldn't compare those two weeks. Please try again."
        return {"response": response, "messages": [AIMessage(content=response)], "needs_clarification": False}

    # --------------------------------------------------
    # 2. Total spending for the requested period.
    # --------------------------------------------------
    if analytics_type == "total_spending":

        try:
            # Use the shared summary function instead of recalculating totals here.
            spending_result = get_spending_summary(
                start_date=start_date,
                end_date=end_date
            )
            total_spending = spending_result["total_spending"]

            response = (
                f"**Your Total Spending — {period_label}**\n\n"
                f"Total spent: **₹{total_spending:,.2f}**\n"
                f"Transactions: **{spending_result['transaction_count']}**"
            )

            # Explain when no expenses were recorded.
            if spending_result["transaction_count"] == 0:
                response += "\n\nNo expenses were recorded during this period."

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:
            print(
                "[TOTAL SPENDING ERROR]",
                str(error),
                flush=True
            )

            response = (
                "I couldn't retrieve your total spending. "
                "Please try again."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

    # --------------------------------------------------
    # Top spending categories for the requested month.
    # --------------------------------------------------
    if analytics_type == "top_categories":
        try:
            # Retrieve all category totals for the resolved date range.
            category_results = get_category_summary(
                start_date=start_date,
                end_date=end_date
            )

            # Sort categories from highest to lowest actual spending.
            ranked_categories = sorted(
                category_results,
                key=lambda row: float(row.get("total_spending") or 0),
                reverse=True
            )

            # Respect an explicit requested count; otherwise show five.
            requested_limit = entities.get("limit", 5)
            try:
                requested_limit = max(1, min(int(requested_limit), 20))
            except (TypeError, ValueError):
                requested_limit = 5
            top_categories = ranked_categories[:requested_limit]
            if not top_categories:
                response = f"No category expenses were recorded in {period_label}."
            else:
                lines = [f"**Top Spending Categories — {period_label}**", ""]
                for index, category in enumerate(top_categories, start=1):
                    amount = float(category.get("total_spending") or 0)
                    lines.append(
                        f"{index}. **{category['category_name']}** — **₹{amount:,.2f}**"
                    )
                response = "\n".join(lines)

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }
        except Exception as error:
            print("[TOP CATEGORIES ERROR]", str(error), flush=True)
            response = "I couldn't retrieve your top spending categories. Please try again."
            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

    # --------------------------------------------------
    # 3. Rank spending categories for one month.
    # --------------------------------------------------
    if analytics_type == "category_rank":

        try:
            from datetime import datetime

            # Retrieve all category totals for the requested month.
            category_results = get_category_summary(
                start_date=start_date,
                end_date=end_date
            )

            # Sort categories from highest to lowest spending.
            ranked_categories = sorted(
                category_results,
                key=lambda row: float(row.get("total_spending") or 0),
                reverse=True
            )

            requested_rank = int(entities.get("rank", 1))

            if not ranked_categories:
                response = (
                    f"No expenses were recorded in {period_label}, "
                    "so I can't rank the spending categories."
                )
            elif requested_rank > len(ranked_categories):
                response = (
                    f"There are only {len(ranked_categories)} categories "
                    f"with spending recorded in {period_label}."
                )
            else:
                selected = ranked_categories[requested_rank - 1]
                amount = float(selected.get("total_spending") or 0)

                response = (
                    f"The {requested_rank}{'nd' if requested_rank == 2 else 'rd' if requested_rank == 3 else 'th' if requested_rank > 3 else 'st'}-highest "
                    f"spending category in {period_label} is "
                    f"**{selected['category_name']}** "
                    f"with **₹{amount:,.2f}**."
                )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:
            print(
                "[CATEGORY RANK ERROR]",
                str(error),
                flush=True
            )
            response = (
                "I couldn't rank your spending categories. Please try again."
            )
            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }


    # --------------------------------------------------
    # 4. Identify the category with the largest month-on-month change.
    # --------------------------------------------------
    if analytics_type == "category_change_analysis":

        try:
            from datetime import datetime

            current_month_value = entities.get("current_month")
            previous_month_value = entities.get("previous_month")

            if not current_month_value or not previous_month_value:
                raise ValueError("Both comparison months are required.")

            current_period = _build_month_period(
                datetime.strptime(current_month_value, "%m-%Y").date()
            )
            previous_period = _build_month_period(
                datetime.strptime(previous_month_value, "%m-%Y").date()
            )

            current_rows = get_category_summary(
                start_date=current_period["start_date"],
                end_date=current_period["end_date"]
            )
            previous_rows = get_category_summary(
                start_date=previous_period["start_date"],
                end_date=previous_period["end_date"]
            )

            current_by_name = {
                row["category_name"]: float(row.get("total_spending") or 0)
                for row in current_rows
            }
            previous_by_name = {
                row["category_name"]: float(row.get("total_spending") or 0)
                for row in previous_rows
            }

            all_categories = set(current_by_name) | set(previous_by_name)
            changes = [
                (
                    category,
                    current_by_name.get(category, 0.0)
                    - previous_by_name.get(category, 0.0)
                )
                for category in all_categories
            ]

            increases = [item for item in changes if item[1] > 0]

            if not changes:
                response = (
                    "No category spending records were found for "
                    f"{previous_period['period_label']} or "
                    f"{current_period['period_label']}."
                )
            elif not increases:
                response = (
                    f"No category increased in {current_period['period_label']} "
                    f"compared with {previous_period['period_label']}."
                )
            else:
                category_name, increase_amount = max(
                    increases,
                    key=lambda item: item[1]
                )
                previous_amount = previous_by_name.get(category_name, 0.0)
                current_amount = current_by_name.get(category_name, 0.0)

                response = (
                    f"**{category_name}** had the largest increase, rising by "
                    f"**₹{increase_amount:,.2f}**.\n\n"
                    f"{previous_period['period_label']}: "
                    f"₹{previous_amount:,.2f}\n\n"
                    f"{current_period['period_label']}: "
                    f"₹{current_amount:,.2f}"
                )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:
            print(
                "[CATEGORY CHANGE ANALYSIS ERROR]",
                str(error),
                flush=True
            )
            response = (
                "I couldn't determine which category changed the most. "
                "Please try again."
            )
            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }


    # --------------------------------------------------
    # 3. Compare category spending percentages across two months.
    # --------------------------------------------------
    if analytics_type == "category_percentage_comparison":

        try:
            # Retrieve category totals and percentages for both months.
            first_period = entities.get("first_period", {})
            second_period = entities.get("second_period", {})

            first_results = get_category_summary(
                start_date=first_period.get("start_date"),
                end_date=first_period.get("end_date")
            )
            second_results = get_category_summary(
                start_date=second_period.get("start_date"),
                end_date=second_period.get("end_date")
            )

            # Index each month's results by category name for comparison.
            first_by_category = {
                row["category_name"]: row for row in first_results
            }
            second_by_category = {
                row["category_name"]: row for row in second_results
            }

            # Include categories found in either month, even if one month
            # has no transactions for that category.
            category_names = sorted(
                set(first_by_category) | set(second_by_category),
                key=str.casefold
            )

            first_label = first_period.get("period_label", "First month")
            second_label = second_period.get("period_label", "Second month")

            response_lines = [
                f"**Category Spending Comparison — {first_label} vs {second_label}**",
                "",
                "| Category | " + first_label + " | " + second_label + " |",
                "|---|---:|---:|"
            ]

            for category_name in category_names:
                first_row = first_by_category.get(category_name, {})
                second_row = second_by_category.get(category_name, {})

                first_amount = float(
                    first_row.get("total_spending") or 0
                )
                second_amount = float(
                    second_row.get("total_spending") or 0
                )
                first_percentage = float(
                    first_row.get("percentage") or 0
                )
                second_percentage = float(
                    second_row.get("percentage") or 0
                )

                response_lines.append(
                    f"| **{category_name}** | "
                    f"₹{first_amount:,.2f} ({first_percentage:.2f}%) | "
                    f"₹{second_amount:,.2f} ({second_percentage:.2f}%) |"
                )

            # Explain when neither month contains category transactions.
            if not category_names:
                response_lines.extend([
                    "",
                    "No expenses were recorded in either month."
                ])

            response = "\n".join(response_lines)

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:
            print(
                "[CATEGORY PERCENTAGE COMPARISON ERROR]",
                str(error),
                flush=True
            )

            response = (
                "I couldn't compare category spending percentages "
                "for those months. Please try again."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

    # --------------------------------------------------
    # Identify the category with the largest share of spending.
    # --------------------------------------------------
    if analytics_type == "category_highest_percentage":
        try:
            result = get_category_analysis(start_date, end_date)
            ranked = sorted(result.get("categories", []), key=lambda item: item.get("percentage", 0), reverse=True)
            if ranked:
                top = ranked[0]
                response = (
                    f"**{top['category_name']}** accounts for the highest share "
                    f"of your spending in {period_label}: **{top['percentage']:.2f}%** "
                    f"(₹{top['total_spending']:,.2f})."
                )
            else:
                response = f"No expenses were recorded in {period_label}."
        except Exception as error:
            print("[CATEGORY RANK ERROR]", str(error), flush=True)
            response = "I couldn't identify your highest-spending category. Please try again."
        return {"response": response, "messages": [AIMessage(content=response)], "needs_clarification": False}

    # --------------------------------------------------
    # 4. Category-wise spending percentages.
    # --------------------------------------------------
    if analytics_type == "category_percentages":

        try:
            # Use the shared category-analysis function for totals and shares.
            category_result = get_category_analysis(
                start_date=start_date,
                end_date=end_date
            )
            category_rows = category_result["categories"]

            if not category_rows:
                response = (
                    f"No expenses were recorded in {period_label}, "
                    "so category percentages are not available."
                )
            else:
                response_lines = [
                    f"**Your Category Spending — {period_label}**",
                    ""
                ]
                for category in category_rows:
                    response_lines.append(
                        f"- **{category['category_name']}**: "
                        f"₹{category['total_spending']:,.2f} "
                        f"({category['percentage']:.2f}%)"
                    )
                response = "\n".join(response_lines)

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:
            print(
                "[CATEGORY PERCENTAGES ERROR]",
                str(error),
                flush=True
            )

            response = (
                "I couldn't retrieve your category percentages. "
                "Please try again."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

    # --------------------------------------------------
    # Find the highest-spending week in the selected period.
    # --------------------------------------------------
    if analytics_type == "weekly_highest":
        try:
            weekly_result = get_weekly_spending(start_date, end_date)
            ranked_weeks = sorted(weekly_result["weeks"], key=lambda week: week["total_spending"], reverse=True)
            if ranked_weeks and ranked_weeks[0]["total_spending"] > 0:
                week = ranked_weeks[0]
                response = (
                    f"Your highest-spending week in {period_label} was "
                    f"**{week['start_date']} to {week['end_date']}**, "
                    f"with **₹{week['total_spending']:,.2f}**."
                )
            else:
                response = f"No expenses were recorded in {period_label}."
        except Exception as error:
            print("[WEEKLY RANK ERROR]", str(error), flush=True)
            response = "I couldn't identify your highest-spending week. Please try again."
        return {"response": response, "messages": [AIMessage(content=response)], "needs_clarification": False}

    # --------------------------------------------------
    # 5. Weekly spending summaries for the requested month.
    # --------------------------------------------------
    if analytics_type == "weekly_summary":

        try:
            from datetime import datetime

            # Use the shared weekly grouping logic for Monday-to-Sunday periods.
            weekly_result = get_weekly_spending(
                start_date=start_date,
                end_date=end_date
            )
            response_lines = [
                f"**Your Weekly Spending — {period_label}**",
                ""
            ]

            for week in weekly_result["weeks"]:
                week_start = datetime.strptime(
                    week["start_date"], "%d-%m-%Y"
                ).date()
                week_end = datetime.strptime(
                    week["end_date"], "%d-%m-%Y"
                ).date()
                response_lines.append(
                    f"- **{week_start.strftime('%d %b')} – "
                    f"{week_end.strftime('%d %b')}**: "
                    f"₹{week['total_spending']:,.2f}"
                )

            if weekly_result["total_spending"] == 0:
                response_lines.extend([
                    "",
                    "No expenses were recorded during this month."
                ])

            response = "\n".join(response_lines)

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:
            print(
                "[WEEKLY SUMMARY ERROR]",
                str(error),
                flush=True
            )

            response = (
                "I couldn't retrieve your weekly summaries. "
                "Please try again."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

    # --------------------------------------------------
    # Handle spending requests involving multiple categories.
    # --------------------------------------------------
    if entities.get("analytics_type") == "multi_category_spending":

        # Retrieve all requested categories and the shared date range.
        category_names = entities.get("category_names", [])
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        period_label = entities.get("period_label")

        try:
            # Retrieve the valid categories and their database IDs.
            categories = get_categories()

            # Store each category's spending for the final response.
            response_lines = [
                f"**Your {period_label} spending by category:**",
                ""
            ]

            # Keep track of the combined spending across these categories.
            combined_spending = 0.0

            # Calculate spending separately for every requested category.
            for category_name in category_names:

                # Find the database category matching the requested name.
                matching_category = next(
                    (
                        category
                        for category in categories
                        if category["category_name"].casefold()
                        == category_name.casefold()
                    ),
                    None
                )

                # Stop if a requested category does not exist.
                if matching_category is None:
                    raise ValueError(
                        f"Category '{category_name}' was not found."
                    )

                # Retrieve transactions for this category and date range.
                category_results = get_category_summary(
                    start_date=start_date,
                    end_date=end_date,
                    category_id=matching_category["category_id"]
                )

                # Use zero when no transactions exist for this category.
                category_spending = (
                    float(category_results[0]["total_spending"])
                    if category_results else 0.0
                )

                # Add this category's amount to the combined total.
                combined_spending += category_spending

                # Prepare a readable result for this category.
                category_line = (
                    f"- **{category_name}:** "
                    f"₹{category_spending:,.2f}"
                )

                # Explicitly identify categories without transactions.
                if category_spending == 0:
                    category_line += " (No transactions recorded)"

                response_lines.append(category_line)

            # Display the combined total for the requested categories.
            response_lines.extend([
                "",
                f"**Total across these categories: "
                f"₹{combined_spending:,.2f}**"
            ])

            # Combine all result lines into one chatbot response.
            response = "\n".join(response_lines)

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:

            # Log the technical error for debugging.
            print(
                "[MULTI-CATEGORY ANALYTICS ERROR]",
                str(error),
                flush=True
            )

            # Return a simple message without exposing database details.
            response = (
                f"I couldn't retrieve the spending breakdown "
                f"for {period_label}. Please try again."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }


    # Check whether this is a single-category spending question.
    if entities.get("analytics_type") == "category_spending":

        # Retrieve the category and requested date range.
        category_name = entities.get("category_name")
        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        period_label = entities.get("period_label")

        try:
            # Retrieve the available expense categories.
            categories = get_categories()

            # Find the database ID matching the requested category.
            matching_category = next(
                (
                    category
                    for category in categories
                    if category["category_name"].casefold()
                    == category_name.casefold()
                ),
                None
            )

            # Stop if the requested category does not exist.
            if matching_category is None:
                raise ValueError(
                    f"Category '{category_name}' was not found."
                )

            # Retrieve spending for this category and month only.
            category_results = get_category_summary(
                start_date=start_date,
                end_date=end_date,
                category_id=matching_category["category_id"]
            )

            # Use zero when no transactions exist for this category.
            total_spending = (
                category_results[0]["total_spending"]
                if category_results else 0
            )

            # Prepare the response using the database result.
            response = (
                f"You spent **₹{float(total_spending):,.2f}** on "
                f"**{category_name}** in **{period_label}**."
            )

            # Explain when no transactions were recorded.
            if total_spending == 0:
                response += (
                    f" No {category_name} transactions were recorded "
                    f"during this month."
                )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        except Exception as error:

            # Log the error without exposing technical details.
            print(
                "[CATEGORY ANALYTICS ERROR]",
                str(error),
                flush=True
            )

            response = (
                f"I couldn't retrieve your {category_name} "
                f"spending for {period_label}."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }


    current_month = entities.get("current_month")
    previous_month = entities.get("previous_month")

    # Retrieve the optional category filter.
    category_name = entities.get("category_name")

    # Check whether both months are available.
    if not current_month or not previous_month:

        response = (
            "Which two months would you like me to compare? "
            "Please provide both months, for example November 2026 "
            "and October 2026."
        )

        return {
            "needs_clarification": True,
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    try:
        # Run the tested monthly comparison function.
        comparison_result = compare_monthly_spending(
        current_month=current_month,
        previous_month=previous_month,
        category_name=category_name
    )

        # Generate the explanation using deterministic calculations.
        response = generate_monthly_comparison_insight(
            comparison_result
        )

        # Return the generated financial explanation.
        return {
            "response": response,
            "messages": [AIMessage(content=response)],
            "needs_clarification": False
        }

    except (ValueError, TypeError, KeyError) as error:

        # Log the error for debugging without exposing technical details.
        print(
            "[ANALYTICS ERROR]",
            str(error),
            flush=True
        )

        response = (
            "I couldn't compare those months using the available "
            "financial records. Please check the months and try again."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    except Exception as error:

        # Handle unexpected database or analytics failures.
        print(
            "[UNEXPECTED ANALYTICS ERROR]",
            str(error),
            flush=True
        )

        response = (
            "Something went wrong while analysing your spending. "
            "Please try again."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    
# --------------------------------------------------
# 4. Search Transactions
# --------------------------------------------------

def search_transactions(**filters):
    """
    Search the financial database using the existing read-only
    expense search tool.

    Keeping this wrapper separate makes it easier to mock database
    results during testing.
    """

    return search_expenses_tool.invoke(filters)



def extract_selection_number(message):
    """
    Extract a numbered or ordinal selection from the user's message.

    Supports:
    - first one
    - the second one
    - edit the first one
    - delete the second one
    - option 2
    - 2
    """

    # Convert the message to lowercase and remove extra spaces.
    message = message.strip().lower()

    # Map ordinal words to their corresponding list positions.
    ordinal_numbers = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5
    }

    # Find ordinal words even when the user includes an action,
    # such as "edit the first one".
    for ordinal, number in ordinal_numbers.items():

        if re.search(
            rf"\b{ordinal}\b(?:\s+one\b)?",
            message
        ):
            return number

    # Support messages such as "edit option 2" or "number 2".
    match = re.search(
        r"\b(?:number|option|choice|no\.?)\s*(\d+)\b",
        message
    )

    if match:
        return int(match.group(1))

    # Support a message containing only a number, such as "2".
    if re.fullmatch(r"\d+", message):
        return int(message)

    # No valid selection was found.
    return None



def classify_approval_response(message):
    """
    Identify whether the user approves, rejects or cancels
    a pending financial operation.

    This function only classifies the response. It does not
    approve or execute any database operation.
    """

    # Normalise the message to make matching consistent.
    message = message.strip().lower()

    # Remove common punctuation and collapse extra spaces.
    message = re.sub(r"[.!?,]+", " ", message)
    message = " ".join(message.split())

    # Recognised approval responses.
    approval_phrases = {
        "yes",
        "yes please",
        "yes confirm",
        "confirm",
        "confirmed",
        "approve",
        "approved",
        "go ahead",
        "do it",
        "apply changes",
        "proceed",
        "proceed please"
    }

    # Recognised rejection responses.
    rejection_phrases = {
        "no",
        "no thanks",
        "no thank you",
        "reject",
        "rejected",
        "do not apply",
        "dont apply",
        "keep the original",
        "leave it unchanged"
    }

    # Recognised cancellation responses.
    cancellation_phrases = {
        "cancel",
        "cancel it",
        "cancel this",
        "cancel the operation",
        "discard",
        "discard it",
        "forget it"
    }

    # Classify only clearly recognised responses.
    if message in approval_phrases:
        return "approved"

    if message in rejection_phrases:
        return "rejected"

    if message in cancellation_phrases:
        return "cancelled"

    # Do not assume approval when the response is ambiguous.
    return "unclear"


def get_latest_user_message(messages):
    """
    Retrieve the latest human message from conversation history.
    """

    # Search backwards so we get the most recent user message.
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    return ""


def detect_simple_budget_query(message):
    """
    Identify straightforward monthly budget questions
    without requiring LLM classification.

    Returns the query type when recognised, otherwise None.
    """

    # Normalise the message for consistent matching.
    message = str(message).strip().lower()

    # Leave requests involving specific or relative months to Groq.
    month_pattern = re.compile(
        r"\b(?:"
        r"january|february|march|april|may|june|"
        r"july|august|september|october|november|december|"
        r"\d{1,2}[-/]\d{4}|"
        r"this month|last month|previous month|next month"
        r")\b",
        re.IGNORECASE
    )

    if month_pattern.search(message):
        return None

    # Detect daily spending guidance requests.
    daily_pattern = re.compile(
        r"\b(?:"
        r"how much can i spend (?:per|a) day|"
        r"daily spending (?:limit|guidance)|"
        r"safe daily spending|"
        r"daily budget guidance"
        r")\b",
        re.IGNORECASE
    )

    if daily_pattern.search(message):
        return "daily_guidance"

    # Detect budget utilisation questions.
    utilisation_pattern = re.compile(
        r"\b(?:"
        r"budget utilisation|"
        r"budget utilization|"
        r"what percentage of my budget|"
        r"how much of my budget have i used|"
        r"percentage of my budget used"
        r")\b",
        re.IGNORECASE
    )

    if utilisation_pattern.search(message):
        return "utilisation"

    # Detect the chatbot's concise "budget status" quick question.
    if re.search(
        r"\b(?:budget\s+status|status\s+of\s+my\s+budget)\b",
        message,
        re.IGNORECASE
    ):
        return "summary"

    # Detect straightforward monthly budget summary requests.
    summary_pattern = re.compile(
        r"\b(?:"
        r"show|display|view|check|get"
        r")\s+(?:my\s+)?(?:monthly\s+)?budget\b|"
        r"\bwhat is my (?:monthly )?budget\b|"
        r"\bhow much budget (?:is )?(?:left|remaining)\b|"
        r"\bhow much is left in my budget\b|"
        r"\bremaining budget\b",
        re.IGNORECASE
    )

    if summary_pattern.search(message):
        return "summary"

    return None


def detect_week_period_comparison_request(message):
    """Detect comparisons between the first and second seven-day blocks of a month."""
    text = str(message).strip().casefold()
    if not re.search(r"\bcompare\b|\bcomparison\b", text):
        return None
    if not re.search(r"\bfirst\s+and\s+second\s+weeks?\b", text):
        return None

    start, _ = _resolve_analytics_period(text)
    first_start = start.replace(day=1)
    first_end = first_start.replace(day=7)
    second_start = first_start.replace(day=8)
    import calendar
    second_end = first_start.replace(day=min(14, calendar.monthrange(start.year, start.month)[1]))

    return {
        "analytics_type": "weekly_period_comparison",
        "first_start_date": first_start.strftime("%d-%m-%Y"),
        "first_end_date": first_end.strftime("%d-%m-%Y"),
        "second_start_date": second_start.strftime("%d-%m-%Y"),
        "second_end_date": second_end.strftime("%d-%m-%Y"),
        "period_label": first_start.strftime("%B %Y"),
    }


def _resolve_analytics_period(message, today=None):
    """Resolve an explicit date range or named calendar month.

    Database dates use DD-MM-YYYY. A range such as
    "between 1 November 2026 and 10 November 2026" is kept exact.
    """
    import calendar
    from datetime import date, datetime, timedelta
    from zoneinfo import ZoneInfo

    text = str(message).strip().casefold()
    today = today or datetime.now(ZoneInfo("Asia/Kolkata")).date()

    # First handle explicit day-month-year ranges so they are not
    # accidentally widened to the whole calendar month.
    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7,
        "july": 7, "aug": 8, "august": 8, "sep": 9,
        "sept": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    month_pattern = (
        r"january|february|march|april|may|june|july|august|"
        r"september|sept|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    )
    range_match = re.search(
        rf"\b(?:between|from)\s+(\d{{1,2}})\s+({month_pattern})"
        rf"(?:\s+(\d{{4}}))?\s+(?:and|to)\s+"
        rf"(\d{{1,2}})\s+({month_pattern})(?:\s+(\d{{4}}))?\b",
        text,
        re.IGNORECASE,
    )
    if range_match:
        d1, m1, y1, d2, m2, y2 = range_match.groups()
        year = int(y1 or y2 or today.year)
        start = date(year, month_names[m1], int(d1))
        end = date(int(y2 or year), month_names[m2], int(d2))
        if start > end:
            raise ValueError("The start date must not be after the end date.")
        return start, end

    # Also support numeric DD-MM-YYYY date ranges.
    numeric_range = re.search(
        r"\b(?:between|from)\s+(\d{1,2})[-/](\d{1,2})[-/](\d{4})"
        r"\s+(?:and|to)\s+(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b",
        text,
    )
    if numeric_range:
        d1, m1, y1, d2, m2, y2 = map(int, numeric_range.groups())
        start, end = date(y1, m1, d1), date(y2, m2, d2)
        if start > end:
            raise ValueError("The start date must not be after the end date.")
        return start, end

    # Resolve named months, relative months, and current month.
    month_start = today.replace(day=1)
    month_match = re.search(
        rf"\b({month_pattern})\s*(\d{{4}})?\b", text, re.IGNORECASE
    )
    if month_match:
        month = month_names[month_match.group(1).casefold()]
        year = int(month_match.group(2) or today.year)
        start = date(year, month, 1)
    elif re.search(r"\b(?:last|previous)\s+month\b", text):
        start = (month_start - timedelta(days=1)).replace(day=1)
    else:
        start = month_start

    # An ongoing month ends today; historical months use their final day.
    if (start.year, start.month) == (today.year, today.month):
        end = today
    else:
        end = start.replace(day=calendar.monthrange(start.year, start.month)[1])
    return start, end




def resolve_analytics_limit_followup(message, previous_intent, previous_entities):
    """Change a ranked-list result count while preserving its active period."""
    if previous_intent != "analytics":
        return None
    previous_entities = previous_entities or {}
    if previous_entities.get("analytics_type") not in {"top_expenses", "top_categories"}:
        return None

    text = str(message or "").strip().casefold()
    count = r"(?P<count>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    patterns = [
        rf"show\s+(?:me\s+)?(?:my\s+)?(?:top\s+)?{count}(?:\s+(?:expenses?|spending\s+categories))?\s+(?:instead|please)",
        rf"(?:show\s+)?(?:my\s+)?top\s+{count}\s+(?:expenses?|spending\s+categories)(?:\s+instead)?",
        rf"(?:make\s+it|change\s+(?:it|the\s+limit)\s+to|set\s+(?:the\s+)?limit\s+to|limit\s+to)\s+{count}",
    ]
    match = None
    for pattern in patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            break
    if not match:
        return None

    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    raw = match.group("count").casefold()
    number = words.get(raw, int(raw) if raw.isdigit() else None)
    if number is None:
        return None

    result = dict(previous_entities)
    result["limit"] = max(1, min(number, 20))
    return result


def resolve_analytics_period_followup(
    message,
    previous_intent,
    previous_entities,
    previous_context=None,
    today=None,
):
    """
    Resolve short month follow-ups against the immediately preceding
    period-based analytics request.

    Supports "last month", "previous month", "the month before that",
    "two months ago", "this month", and named months. The active analytics
    type, category, rank and result limit are preserved.
    """
    from datetime import date, datetime, timedelta
    import calendar

    if previous_intent != "analytics":
        return None

    previous_entities = previous_entities or {}
    previous_context = previous_context or {}
    analytics_type = previous_entities.get("analytics_type")
    if not analytics_type:
        return None

    # These analytics return a result for one selected period and can be
    # meaningfully rerun for another month.
    supported_types = {
        "top_categories",
        "category_rank",
        "category_spending",
        "multi_category_spending",
        "total_spending",
        "average_daily_spending",
        "weekly_summary",
        "weekly_highest",
        "spending_days",
        "category_percentages",
        "category_highest_percentage",
        "payment_method_analysis",
        "top_expenses",
        "category_change_analysis",
        "category_percentage_comparison",
        "category_budget_status",
    }
    if analytics_type not in supported_types:
        return None

    text = str(message or "").strip().casefold()
    prefix = r"(?:(?:and|what about|how about|then|for)\s+)?"
    period_pattern = re.fullmatch(
        prefix
        + r"(?:the\s+)?"
        + r"(month before that|month before|two months ago|"
          r"2 months ago|last month|previous month|this month|"
          r"current month|last|previous|current|this|"
          r"january|february|march|april|may|june|july|august|"
          r"september|sept|october|november|december|"
          r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        + r"(?:\s+(\d{4}))?[?.!\s]*",
        text,
        re.IGNORECASE,
    )
    if not period_pattern:
        return None

    today = today or date.today()
    phrase = period_pattern.group(1)
    explicit_year = period_pattern.group(2)
    month_start = today.replace(day=1)

    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    if phrase in {"month before that", "month before"}:
        # "That" refers to the period returned immediately before this turn.
        prior_start = previous_entities.get("start_date")
        if not prior_start:
            prior_start = (previous_context.get("period") or {}).get("start_date")
        if not prior_start and previous_entities.get("budget_month"):
            try:
                prior_start = datetime.strptime(
                    previous_entities["budget_month"], "%m-%Y"
                ).date().strftime("%d-%m-%Y")
            except (TypeError, ValueError):
                return None
        if not prior_start:
            return None
        try:
            reference = date.fromisoformat(prior_start) if "-" in prior_start and len(prior_start) == 10 and prior_start[4] == "-" else date.fromisoformat(
                f"{prior_start[6:10]}-{prior_start[3:5]}-{prior_start[0:2]}"
            )
        except (TypeError, ValueError):
            return None
        target = reference.replace(day=1) - timedelta(days=1)
        start = target.replace(day=1)
    elif phrase in {"two months ago", "2 months ago"}:
        target = month_start
        for _ in range(2):
            target = target - timedelta(days=1)
            target = target.replace(day=1)
        start = target
    elif phrase in {"last month", "previous month", "last", "previous"}:
        start = (month_start - timedelta(days=1)).replace(day=1)
    elif phrase in {"this month", "current month", "this", "current"}:
        start = month_start
    else:
        start = date(int(explicit_year or today.year), month_names[phrase], 1)

    last_day = calendar.monthrange(start.year, start.month)[1]
    end = (
        today if (start.year, start.month) == (today.year, today.month)
        else start.replace(day=last_day)
    )

    result = dict(previous_entities)
    result["start_date"] = start.strftime("%d-%m-%Y")
    result["end_date"] = end.strftime("%d-%m-%Y")
    result["period_label"] = start.strftime("%B %Y")

    # Budget status uses MM-YYYY rather than date boundaries.
    if analytics_type == "category_budget_status":
        result["budget_month"] = start.strftime("%m-%Y")

    return result

def detect_period_analytics_request(message):
    """Detect deterministic, single-period spending analytics requests."""
    message = str(message).strip().casefold()

    patterns = {
        "average_daily_spending": r"\b(?:average\s+daily\s+spending|average\s+daily\s+expenses?|daily\s+average\s+spending|average\s+spend(?:ing)?\s+per\s+day)\b",
        "spending_days": r"\b(?:how\s+many\s+days\s+(?:did\s+i|have\s+i)\s+(?:actually\s+)?spend\s+money|days\s+(?:i\s+)?spent\s+money|active\s+spending\s+days)\b",
        "weekly_summary": r"\b(?:weekly\s+spending|weekly\s+summary|spending\s+week\s+by\s+week|break\s+down\s+my\s+spending\s+week\s+by\s+week)\b",
        "weekly_highest": r"\b(?:which|what)\s+week\s+(?:did\s+i\s+spend\s+the\s+most|had\s+the\s+highest\s+spending|was\s+the\s+most\s+expensive)\b",
        "category_highest_percentage": r"\b(?:which|what)\s+categor(?:y|ies)\s+(?:accounts?\s+for|has|had)\s+(?:the\s+)?highest\s+percentage\b|\bhighest\s+percentage\s+of\s+(?:my\s+)?spending\b",
        "category_percentages": r"\b(?:category|categories)[\s-]+(?:wise[\s-]+)?spending\b|\bspending\s+by\s+categor(?:y|ies)\b|\bcategory\s+breakdown\b|\bcategory[\s-]+percentages\b",
        "payment_method_analysis": r"\b(?:payment\s+methods?|payment\s+method\s+analysis|spending\s+by\s+payment\s+methods?)\b",
        "top_expenses": r"\b(?:top|largest|biggest|highest)(?:\s+\d{1,2})?\s+expenses?\b|\bmost\s+expensive\s+transactions?\b",
        "monthly_trend": r"\bmonthly\s+(?:spending\s+)?trends?\b|\bspending\s+trend\b",
        "total_spending": r"\b(?:overall\s+spending|overall\s+expenses|show\s+(?:me\s+)?(?:my\s+)?spending|show\s+(?:me\s+)?(?:my\s+)?expenses|total\s+spending|total\s+spent|total\s+expenses|how\s+much\s+(?:have|did)\s+i\s+spent|how\s+much\s+(?:have|did)\s+i\s+spend)\b",
    }

    # More specific patterns must be checked before generic totals.
    analytics_type = next(
        (kind for kind, pattern in patterns.items() if re.search(pattern, message)),
        None,
    )
    if analytics_type is None:
        return None

    start, end = _resolve_analytics_period(message)
    period_label = start.strftime("%B %Y")

    # Trend requests span the latest twelve months, ending today.
    if analytics_type == "monthly_trend":
        from datetime import timedelta
        month_index = end.year * 12 + end.month - 1 - 11
        year, month_index = divmod(month_index, 12)
        start = start.replace(year=year, month=month_index + 1, day=1)
        end = _resolve_analytics_period("this month")[1]
        period_label = f"{start.strftime('%B %Y')} – {end.strftime('%B %Y')}"

    request = {
        "analytics_type": analytics_type,
        "start_date": start.strftime("%d-%m-%Y"),
        "end_date": end.strftime("%d-%m-%Y"),
        "period_label": period_label,
    }

    # Extract the requested count for ranked expense lists; default to five.
    if analytics_type == "top_expenses":
        count_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        count_match = re.search(
            r"\b(?:top|largest|biggest|highest)\s+"
            r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+expenses?\b",
            message,
            re.IGNORECASE,
        )
        raw_count = count_match.group(1).casefold() if count_match else None
        requested_limit = (
            count_words.get(raw_count)
            if raw_count and not raw_count.isdigit()
            else int(raw_count) if raw_count else 5
        )
        request["limit"] = max(1, min(requested_limit or 5, 20))
    return request


def detect_average_daily_followup(
    message,
    previous_intent,
    previous_entities
):
    """
    Resolve month follow-ups after an average daily spending question.

    Examples:
    - What about last month?
    - What about this month?
    - What about October 2026?
    """

    # Only handle follow-ups to average daily spending requests.
    if (
        previous_intent != "analytics"
        or previous_entities.get("analytics_type")
        != "average_daily_spending"
    ):
        return None

    # Normalise the follow-up message.
    message = str(message).strip().casefold()

    # Recognise short month follow-up phrases.
    followup_pattern = re.fullmatch(
        r"(?:and\s+|what\s+about\s+|how\s+about\s+)?"
        r"(this\s+month|current\s+month|last\s+month|"
        r"previous\s+month|"
        r"january|february|march|april|may|june|july|"
        r"august|september|sept|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"(?:\s+(\d{4}))?"
        r"[?.!\s]*",
        message,
        re.IGNORECASE
    )

    # Do not intercept unrelated messages.
    if not followup_pattern:
        return None

    # Resolve the requested period using the existing detector.
    requested_period = followup_pattern.group(1)

    # Reuse the average-spending detector to resolve dates.
    resolved_request = detect_period_analytics_request(
        f"average daily spending in {requested_period}"
        + (
            f" {followup_pattern.group(2)}"
            if followup_pattern.group(2)
            else ""
        )
    )

    # Return the resolved period for the analytics node.
    return resolved_request

def detect_weekly_summary_followup(
    message,
    previous_intent,
    previous_entities
):
    """
    Resolve month-only follow-ups when the previous request
    was a weekly spending summary.

    Examples:
    - What about last month? -> Previous calendar month.
    - What about this month? -> Current calendar month.
    - What about the month before that? -> Month before
      the month in the previous weekly-summary request.
    """

    # Only continue an existing weekly-summary conversation.
    if (
        previous_intent != "analytics"
        or previous_entities.get("analytics_type") != "weekly_summary"
    ):
        return None

    # Normalise the latest user message.
    message = str(message).strip().casefold()

    # Recognise month-only follow-up phrases.
    if re.fullmatch(
        r"(?:and\s+|what\s+about\s+|how\s+about\s+)?"
        r"(?:last\s+month|previous\s+month)"
        r"[?.!\s]*",
        message
    ):
        month_offset = -1

    elif re.fullmatch(
        r"(?:and\s+|what\s+about\s+|how\s+about\s+)?"
        r"(?:this\s+month|current\s+month)"
        r"[?.!\s]*",
        message
    ):
        month_offset = 0

    elif re.fullmatch(
        r"(?:and\s+|what\s+about\s+|how\s+about\s+)?"
        r"(?:the\s+)?month\s+before\s+that"
        r"[?.!\s]*",
        message
    ):
        month_offset = -2

    else:
        return None

    # Use the Indian local date for calendar-month calculations.
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    # Start with the first day of the current month.
    target_month = today.replace(day=1)

    if month_offset == -1:
        # Move to the previous calendar month.
        target_month = (
            target_month - timedelta(days=1)
        ).replace(day=1)

    elif month_offset == -2:
        # Resolve the month before the previous weekly-summary period.
        previous_start = previous_entities.get("start_date")

        if not previous_start:
            return None

        try:
            previous_month_start = datetime.strptime(
                previous_start,
                "%d-%m-%Y"
            ).date().replace(day=1)

            target_month = (
                previous_month_start - timedelta(days=1)
            ).replace(day=1)

        except (ValueError, TypeError):
            return None

    # Calculate the first day of the following month.
    if target_month.month == 12:
        next_month = target_month.replace(
            year=target_month.year + 1,
            month=1
        )
    else:
        next_month = target_month.replace(
            month=target_month.month + 1
        )

    # The last day of the requested month is one day before
    # the following month begins.
    month_end = next_month - timedelta(days=1)

    # Return the same analytics type with the requested full month.
    return {
        "analytics_type": "weekly_summary",
        "start_date": target_month.strftime("%d-%m-%Y"),
        "end_date": month_end.strftime("%d-%m-%Y"),
        "period_label": target_month.strftime("%B %Y")
    }


def detect_monthly_spending_followup(
    message,
    previous_intent,
    previous_entities
):
    """
    Resolve a month-only follow-up using the period analysed
    in the previous analytics response.

    Example:
    September 2026 -> "What about August?" -> August 2026.
    """

    # Only continue an existing analytics conversation.
    if previous_intent != "analytics":
        return None

    # Check that the previous request contains a period.
    previous_label = previous_entities.get("period_label")
    previous_start = previous_entities.get("start_date")

    if not previous_label and not previous_start:
        return None

    # Recognise a month name in a short follow-up.
    message = str(message).strip()

    month_match = re.fullmatch(
        r"(?:(?:and|what about|how about)\s+)?"
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
        r"[?.!\s]*",
        message,
        re.IGNORECASE
    )

    if not month_match:
        return None

    from datetime import datetime
    from calendar import monthrange

    # Get the year from the previously analysed period.
    try:
        if previous_label:
            previous_date = datetime.strptime(
                previous_label,
                "%B %Y"
            ).date()
        else:
            previous_date = datetime.strptime(
                previous_start,
                "%d-%m-%Y"
            ).date()

        # Resolve the requested month name.
        requested_month = datetime.strptime(
            month_match.group(1)[:3].title(),
            "%b"
        ).month

        # Keep the previous period's year unless the requested month
        # is later than the previous month, indicating the prior year.
        target_year = previous_date.year

        if requested_month > previous_date.month:
            target_year -= 1

        # Build the complete calendar-month date range.
        first_day = datetime(
            target_year,
            requested_month,
            1
        ).date()

        last_day = monthrange(
            target_year,
            requested_month
        )[1]

        last_date = first_day.replace(day=last_day)

        return {
            "analytics_type": "total_spending",
            "start_date": first_day.strftime("%d-%m-%Y"),
            "end_date": last_date.strftime("%d-%m-%Y"),
            "period_label": first_day.strftime("%B %Y")
        }

    except (ValueError, TypeError, KeyError) as error:
        print(
            "[DEBUG] Monthly follow-up could not be resolved:",
            str(error),
            flush=True
        )
        return None

def detect_category_percentage_comparison(message):
    """
    Detect category-percentage comparisons across two calendar months.

    Supports examples such as:
    - Compare category spending for November 2026 and October 2026.
    - Show category spending percentages for Nov and Oct 2026.
    - Compare category spending for November 2026 and the month before that.
    """

    from datetime import date, timedelta
    import calendar

    # Only handle messages that clearly request a category comparison.
    message_text = str(message).strip().casefold()
    comparison_pattern = re.compile(
        r"\b(?:compare|comparison|versus|vs\.?|between)\b",
        re.IGNORECASE
    )
    category_pattern = re.compile(
        r"\b(?:categor(?:y|ies)\s+(?:spending|expenses|breakdown|percentages?)|"
        r"spending\s+by\s+categor(?:y|ies)|"
        r"category[- ]wise\s+spending)\b",
        re.IGNORECASE
    )

    # Also accept "show category spending ... and ..." as a comparison.
    has_two_periods = bool(
        re.search(r"\band\b", message_text)
        or re.search(r"\bmonth\s+before\s+that\b", message_text)
    )
    if not category_pattern.search(message_text) or not (
        comparison_pattern.search(message_text) or has_two_periods
    ):
        return None

    # Map full and abbreviated month names to calendar month numbers.
    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9,
        "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12
    }

    # Locate month names and any year written immediately after them.
    month_pattern = re.compile(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|sept|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"\s*(\d{4})?\b",
        re.IGNORECASE
    )
    matches = list(month_pattern.finditer(message_text))

    # Use the Indian local date when the user omits a year.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    months = []
    for match in matches:
        month_number = month_names[match.group(1)]
        year_text = match.group(2)
        months.append((month_number, int(year_text) if year_text else None))

    # A comparison needs one explicitly named month and a relative second
    # month, or two named months.
    if len(months) >= 2:
        first_month, first_year = months[0]
        second_month, second_year = months[1]

        # If only one year is supplied, apply it to both month references.
        shared_year = first_year or second_year or today.year
        first_date = date(first_year or shared_year, first_month, 1)
        second_date = date(second_year or shared_year, second_month, 1)

    elif len(months) == 1 and re.search(
        r"\b(?:the\s+)?month\s+before\s+that\b",
        message_text
    ):
        first_month, first_year = months[0]
        first_date = date(first_year or today.year, first_month, 1)

        # Resolve the month immediately before the named month.
        previous_month_end = first_date - timedelta(days=1)
        second_date = previous_month_end.replace(day=1)

    else:
        return None

    # Calculate complete month boundaries in the database date format.
    def month_range(month_date):
        last_day = calendar.monthrange(
            month_date.year,
            month_date.month
        )[1]
        start = month_date.replace(day=1)
        end = month_date.replace(day=last_day)
        return {
            "start_date": start.strftime("%d-%m-%Y"),
            "end_date": end.strftime("%d-%m-%Y"),
            "period_label": start.strftime("%B %Y")
        }

    first_range = month_range(first_date)
    second_range = month_range(second_date)

    return {
        "analytics_type": "category_percentage_comparison",
        "first_period": first_range,
        "second_period": second_range
    }


def detect_top_categories_request(message):
    """Detect requests for a ranked list of spending categories.

    An omitted period defaults through _resolve_analytics_period() to the
    current month. Explicit months and date ranges remain unchanged.
    """
    text = str(message).strip().casefold()

    # Match phrases such as "top categories", "top 3 spending categories",
    # and "show my five highest-spending categories".
    match = re.search(
        r"\b(?:top|highest|largest|biggest)\s+"
        r"(?:(?P<count>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
        r"(?:spending\s+)?categor(?:y|ies)\b|"
        r"\b(?:show|list|rank)\s+(?:me\s+)?(?:my\s+)?"
        r"(?:(?P<count_alt>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
        r"(?:top|highest|largest|biggest)\s+"
        r"(?:spending\s+)?categor(?:y|ies)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None

    count_text = match.group("count") or match.group("count_alt")
    count_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    limit = count_words.get(count_text, int(count_text) if count_text and count_text.isdigit() else 5)
    limit = max(1, min(limit, 20))

    # Explicit dates/ranges take precedence; otherwise this resolves to
    # the current month and runs through today.
    start, end = _resolve_analytics_period(text)
    period_label = (
        start.strftime("%B %Y") if (start.year, start.month) == (end.year, end.month)
        else f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}"
    )
    return {
        "analytics_type": "top_categories",
        "limit": limit,
        "start_date": start.strftime("%d-%m-%Y"),
        "end_date": end.strftime("%d-%m-%Y"),
        "period_label": period_label,
    }


def detect_category_period_request(message):
    """Detect category-wise spending requests for a selected period."""
    text = str(message).strip().casefold()
    category_pattern = re.compile(
        r"\b(?:"
        r"categor(?:y|ies)\s+(?:spending|expenses|breakdown|percentages?)|"
        r"spending\s+by\s+categor(?:y|ies)|"
        r"expenses\s+by\s+categor(?:y|ies)|"
        r"category[- ]wise\s+spending|"
        r"categories\s+spending"
        r")\b",
        re.IGNORECASE,
    )
    short_followup_pattern = re.compile(
        r"^\s*(?:and\s+|what\s+about\s+|how\s+about\s+)?"
        r"(?:my\s+)?categor(?:y|ies)\s+spending[?.!\s]*$",
        re.IGNORECASE,
    )
    if not (category_pattern.search(text) or short_followup_pattern.fullmatch(text)):
        return None

    # Keep the category breakdown period consistent with other analytics:
    # current month through today, full calendar month for historical months.
    start, end = _resolve_analytics_period(text)
    return {
        "analytics_type": "category_percentages",
        "start_date": start.strftime("%d-%m-%Y"),
        "end_date": end.strftime("%d-%m-%Y"),
        "period_label": start.strftime("%B %Y"),
    }


def detect_analytics_category(message):
    """
    Identify an existing expense category mentioned in an
    analytics request, using the database category names.
    """

    # Retrieve valid categories rather than hardcoding names.
    categories = get_categories()

    message = str(message).casefold()

    # Match longer category names first to avoid partial matches.
    categories = sorted(
        categories,
        key=lambda item: len(item["category_name"]),
        reverse=True
    )

    for category in categories:
        name = category["category_name"]

        # Match the complete category name, ignoring case.
        if re.search(
            rf"(?<!\w){re.escape(name.casefold())}(?!\w)",
            message
        ):
            return name

    return None


def detect_analytics_categories(message):
    """
    Find every valid expense category mentioned in a message.

    Returns category names in the order they appear in the message.
    This supports questions such as:
    "What about Transport, Entertainment and Food?"
    """

    # Retrieve the categories that actually exist in the database.
    categories = get_categories()

    # Normalise the message for case-insensitive matching.
    message = str(message).casefold()

    # Store each matched category with its position in the message.
    matched_categories = []

    for category in categories:
        # Retrieve the actual category name from the database.
        name = category["category_name"]

        # Find the complete category name in the user's message.
        match = re.search(
            rf"(?<!\w){re.escape(name.casefold())}(?!\w)",
            message
        )

        # Keep the category and its position when it is mentioned.
        if match:
            matched_categories.append(
                (match.start(), name)
            )

    # Sort categories by where they appear in the user's message.
    matched_categories.sort(key=lambda item: item[0])

    # Return unique category names in their original order.
    return list(dict.fromkeys(
        name for _, name in matched_categories
    ))


def detect_category_spending_request(
    message,
    previous_category=None,
    previous_period=None
):
    """
    Detect category spending questions and natural follow-ups.

    Examples:
    - How much did I spend on Food last month?
    - What about Utilities?
    - And this month?
    """

    # Detect a category explicitly mentioned in the latest message.
    category_name = detect_analytics_category(message)

    # Retain the previous category when the user does not name another one.
    if not category_name:
        category_name = previous_category

    # A category is required to perform this analysis.
    if not category_name:
        return None

    # Normalise the message for consistent phrase matching.
    message_lower = str(message).strip().casefold()

    # Recognise full spending questions and short follow-up messages.
    spending_pattern = re.compile(
        r"\b(?:"
        r"how\s+much\s+did\s+i\s+spend|"
        r"how\s+much\s+have\s+i\s+spent|"
        r"what\s+did\s+i\s+spend|"
        r"how\s+much\s+did\s+i\s+spend\s+on\s+(?:that|this|it)|"
        r"how\s+much\s+have\s+i\s+spent\s+on\s+(?:that|this|it)|"
        r"show\s+(?:me\s+)?(?:my\s+)?spending|"
        r"show\s+(?:me\s+)?(?:my\s+)?expenses|"
        r"what\s+about|"
        r"how\s+about"
        r")\b",
        re.IGNORECASE
    )

    
    # Recognise period-only follow-ups such as "And this month?"
    period_followup_pattern = re.compile(
        r"^\s*(?:and\s+|what\s+about\s+|how\s+about\s+)?"
        r"(?:this\s+month|this|last\s+month|last|"
        r"current\s+month|current|previous\s+month|previous)"
        r"\s*[?.!]*\s*$",
        re.IGNORECASE
    )

    # Recognise category-only follow-ups such as "And Food?"
    category_only_followup_pattern = re.compile(
        r"^\s*(?:and\s+)?"
        + re.escape(category_name.casefold())
        + r"\s*[?.!]*\s*$",
        re.IGNORECASE
    )


    # Recognise requests such as "Show my Utilities spending".
    category_spending_pattern = re.compile(
        r"\b(?:show|display|view|get|check|what\s+is|"
        r"what\s+was|tell\s+me)\s+"
        r"(?:me\s+)?(?:my\s+)?"
        + re.escape(category_name.casefold())
        + r"\s+(?:spending|expenses|expense|expenditure)\b",
        re.IGNORECASE
    )


    # Continue when the message contains a recognised analytics request
    # or a valid category/period follow-up.
    if not (
        spending_pattern.search(message_lower)
        or category_spending_pattern.search(message_lower)
        or period_followup_pattern.fullmatch(message_lower)
        or category_only_followup_pattern.fullmatch(message_lower)
        or re.fullmatch(
            r"(?:how much did i spend on|how much have i spent on)\s+(?:that|this|it)\??",
            message_lower,
            re.IGNORECASE
        )
    ):
        return None


    # Reuse the same date parser as every other analytics request.
    # This supports explicit months, exact date ranges and relative months.
    try:
        start, end = _resolve_analytics_period(message_lower)
    except ValueError:
        return None

    # A category-only follow-up retains the previous period unless the
    # user supplied a new period in this message.
    has_period = bool(
        re.search(r"\b(?:this|current|last|previous)\s+month\b", message_lower)
        or re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", message_lower)
        or re.search(r"\b(?:between|from)\s+\d{1,2}\b", message_lower)
    )
    if previous_period and not has_period:
        start_text = previous_period.get("start_date")
        end_text = previous_period.get("end_date")
        label = previous_period.get("period_label")
    else:
        start_text = start.strftime("%d-%m-%Y")
        end_text = end.strftime("%d-%m-%Y")
        label = start.strftime("%B %Y")

    return {
        "analytics_type": "category_spending",
        "category_name": category_name,
        "start_date": start_text,
        "end_date": end_text,
        "period_label": label
    }




def _build_month_period(month_date):
    """Build a complete calendar-month date range for analytics."""
    import calendar

    first_day = month_date.replace(day=1)
    last_day_number = calendar.monthrange(
        first_day.year,
        first_day.month
    )[1]
    last_day = first_day.replace(day=last_day_number)

    return {
        "start_date": first_day.strftime("%d-%m-%Y"),
        "end_date": last_day.strftime("%d-%m-%Y"),
        "period_label": first_day.strftime("%B %Y")
    }


def detect_category_rank_request(message):
    """Detect the highest/ordinal spending category for a requested period."""
    text = str(message).strip().casefold()

    rank_match = re.search(
        r"\b(?:(?P<ordinal>first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)"
        r"\s*(?:[- ]highest|[- ]largest|[- ]biggest)|"
        r"(?P<highest>highest|largest|biggest))\s+"
        r"(?:spending\s+)?categor(?:y|ies)\b|"
        r"\b(?:highest|largest|biggest)\s+spending\s+categor(?:y|ies)\b",
        text,
        re.IGNORECASE
    )

    # Natural-language questions that ask for the leading category without
    # using the phrase "highest spending category".
    natural_rank_pattern = re.compile(
        r"\b(?:which|what)\s+categor(?:y|ies)\b.*"
        r"\b(?:did\s+i\s+spend\s+the\s+most\s+on|"
        r"contributed\s+the\s+most|had\s+the\s+most\s+spending|"
        r"accounts?\s+for\s+the\s+most|"
        r"has?\s+the\s+largest\s+share|"
        r"costs?\s+me\s+the\s+most|"
        r"is\s+the\s+most\s+expensive)\b|"
        r"\bmost\s+spent\s+categor(?:y|ies)\b",
        re.IGNORECASE
    )

    if not rank_match and not natural_rank_pattern.search(text):
        return None

    ordinal = rank_match.group("ordinal") if rank_match else None
    rank_values = {
        "first": 1, "1st": 1, "second": 2, "2nd": 2,
        "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5
    }
    rank = rank_values.get(ordinal, 1)

    # Use the shared parser so "this month", "current", named months,
    # and explicit date ranges resolve consistently across analytics.
    start, end = _resolve_analytics_period(text)
    return {
        "analytics_type": "category_rank",
        "rank": rank,
        "start_date": start.strftime("%d-%m-%Y"),
        "end_date": end.strftime("%d-%m-%Y"),
        "period_label": (
            start.strftime("%B %Y") if start.month == end.month
            and start.year == end.year
            else f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}"
        )
    }


def detect_category_change_request(message):
    """
    Detect which category increased/decreased most between two months.

    If the user does not specify periods, compare this month with last month.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    text = str(message).strip().casefold()
    change_pattern = re.compile(
        r"\b(?:which|what)\s+categor(?:y|ies)\b.*"
        r"\b(?:increased|rose|grew|went\s+up|decreased|fell|dropped)\b|"
        r"\b(?:categor(?:y|ies))\b.*"
        r"\b(?:increased|rose|grew|went\s+up|decreased|fell|dropped)\b|"
        r"\b(?:biggest|largest|most)\s+(?:category\s+)?"
        r"(?:increase|rise|growth|decrease|drop)\b",
        re.IGNORECASE
    )

    if not change_pattern.search(text):
        return None

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    current_month = today.replace(day=1)
    previous_month = (current_month - timedelta(days=1)).replace(day=1)

    return {
        "analytics_type": "category_change_analysis",
        "current_month": current_month.strftime("%m-%Y"),
        "previous_month": previous_month.strftime("%m-%Y")
    }


def detect_relative_analytics_months(message, is_analytics_followup=False):
    """
    Resolve relative month comparisons using the current Indian date.

    Examples:
    - Compare this month with last month.
    - Compare current month with previous month.
    - current and earlier (when answering an analytics clarification).

    Returns the two months in MM-YYYY format, or None.
    """

    # Normalise the message before checking its wording.
    message = str(message).strip().lower()

    # Identify references to the current month.
    current_pattern = re.compile(
        r"\b(?:this month|current month|current)\b",
        re.IGNORECASE
    )

    # Identify references to the preceding month.
    previous_pattern = re.compile(
        r"\b(?:last(?:\s+month)?|previous(?:\s+month)?|"
        r"prior(?:\s+month)?|earlier(?:\s+month)?)\b",
        re.IGNORECASE
    )

    
    # Check whether the message contains current and previous month references.
    has_current = bool(current_pattern.search(message))
    has_previous = bool(previous_pattern.search(message))

    # Recognise requests for significant category comparisons.
    # Do not treat every question beginning with "which category" as a
    # month-on-month comparison. Single-period ranking questions are handled
    # by detect_category_rank_request() and should default to this month.
    significant_category_pattern = re.compile(
        r"\b(?:compare|comparison|change|increased|decreased)\b.*"
        r"\bcategor(?:y|ies)\b|"
        r"\b(?:categor(?:y|ies))\b.*"
        r"\b(?:compare|comparison|increased|decreased)\b",
        re.IGNORECASE
    )

    is_significant_category_request = bool(
        significant_category_pattern.search(message)
    )

    # Only explicit comparison language should default to two months.
    if is_significant_category_request:
        has_current = True
        has_previous = True


    # A short reply such as "current and earlier" is meaningful
    # only when the chatbot has already asked for months to compare.
    if not (has_current and has_previous):
        if not is_analytics_followup:
            return None

        # Recognise short clarification replies.
        short_reply_pattern = re.compile(
            r"\bcurrent\s+and\s+(?:the\s+)?earlier\b|"
            r"\bthis\s+and\s+(?:the\s+)?previous\b|"
            r"\bcurrent\s+and\s+(?:the\s+)?previous\b|"
            r"\bthis\s+month\s+and\s+(?:the\s+)?earlier\b",
            re.IGNORECASE
        )

        if not short_reply_pattern.search(message):
            return None

    # Use the current date in the user's Indian timezone.
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    # Format the current month as MM-YYYY.
    current_month = today.strftime("%m-%Y")

    # The day before the first of this month belongs to the previous month.
    previous_month_date = (
        today.replace(day=1) - timedelta(days=1)
    )

    # Format the previous month as MM-YYYY.
    previous_month = previous_month_date.strftime("%m-%Y")

    # Return the two months for the analytics workflow.
    return {
        "current_month": current_month,
        "previous_month": previous_month
    }


def detect_hypothetical_spending(message):
    """
    Detect an explicitly hypothetical spending question and extract
    the additional amount without relying on the LLM.

    Returns the amount as a float, or None if the message is not
    a recognised hypothetical spending scenario.
    """

    # Normalise the message for consistent phrase matching.
    message = str(message).strip().lower()

    # Recognise phrases that clearly indicate hypothetical spending.
    scenario_pattern = re.compile(
        r"\b(?:"
        r"what\s+if\s+i\s+(?:were\s+to\s+)?spend|"
        r"what\s+happens?\s+if\s+i\s+spend|"
        r"if\s+i\s+spend|"
        r"suppose\s+i\s+spend|"
        r"imagine\s+(?:i\s+)?spending"
        r")\b",
        re.IGNORECASE
    )

    # Do not override requests that explicitly ask to record an expense.
    recording_pattern = re.compile(
        r"\b(?:add|record|log|save)\b.{0,40}"
        r"\b(?:expense|transaction)\b",
        re.IGNORECASE
    )

    # Continue only when the message clearly describes a scenario.
    scenario_match = scenario_pattern.search(message)

    if not scenario_match or recording_pattern.search(message):
        return None

    # Look for a numeric amount after the hypothetical phrase.
    amount_pattern = re.compile(
        r"(?:₹|rs\.?\s*)?"
        r"(\d[\d,]*(?:\.\d+)?\s*[kK]?)",
        re.IGNORECASE
    )

    amount_match = amount_pattern.search(
        message[scenario_match.end():]
    )

    # An amount is required to calculate the scenario.
    if not amount_match:
        return None

    # Remove commas and whitespace before converting the amount.
    raw_amount = amount_match.group(1).replace(",", "").strip()

    # Support shorthand amounts such as 5k and 10k.
    multiplier = 1000 if raw_amount.lower().endswith("k") else 1

    if multiplier == 1000:
        raw_amount = raw_amount[:-1].strip()

    try:
        # Convert the extracted amount into a numeric value.
        amount = float(raw_amount) * multiplier

        # Reject invalid, infinite or negative amounts.
        if not math.isfinite(amount) or amount < 0:
            return None

        return amount

    except (ValueError, TypeError):
        return None

def is_hypothetical_reset_request(message):
    """
    Detect requests to clear the running hypothetical scenario.
    This does not modify actual transactions or budget records.
    """

    # Normalise the message before checking reset phrases.
    message = str(message).strip().lower()

    reset_pattern = re.compile(
        r"\b(?:"
        r"start\s+(?:a\s+)?new\s+scenario|"
        r"new\s+scenario|"
        r"reset\s+(?:the\s+)?scenario|"
        r"clear\s+(?:the\s+)?scenario|"
        r"ignore\s+(?:the\s+)?previous\s+(?:scenario|hypothetical\s+spending)|"
        r"forget\s+(?:the\s+)?previous\s+(?:scenario|hypothetical\s+spending)"
        r")\b",
        re.IGNORECASE
    )

    return bool(reset_pattern.search(message))

def detect_amount_only(message):
    """
    Extract an amount when the user replies with only a number,
    such as '5000', '₹5,000' or '5k'.
    """

    # Normalise the user's reply.
    message = str(message).strip().lower()

    # Accept only a standalone amount, not a sentence.
    amount_pattern = re.compile(
        r"(?:₹|rs\.?\s*)?"
        r"(\d[\d,]*(?:\.\d+)?\s*[kK]?)",
        re.IGNORECASE
    )

    match = amount_pattern.fullmatch(message)

    if not match:
        return None

    # Remove commas and whitespace before conversion.
    raw_amount = match.group(1).replace(",", "").strip()

    # Convert shorthand amounts such as 5k into 5000.
    multiplier = 1000 if raw_amount.lower().endswith("k") else 1

    if multiplier == 1000:
        raw_amount = raw_amount[:-1].strip()

    try:
        amount = float(raw_amount) * multiplier

        if not math.isfinite(amount) or amount < 0:
            return None

        return amount

    except (ValueError, TypeError):
        return None

   
def format_transaction(transaction):
    """
    Format a transaction so the chatbot can display its details
    consistently.
    """

    # Read the transaction fields returned by the database.
    transaction_id = transaction.get("expense_id")
    expense_name = normalize_expense_name(
        str(transaction.get("expense_name", "Unknown"))
    )
    amount = transaction.get("amount", 0)
    expense_date = transaction.get("expense_date", "Unknown")
    category = transaction.get(
        "category_name",
        transaction.get("category", "Unknown")
    )
    raw_payment_method = transaction.get("payment_method", "Unknown")
    payment_method = (
        normalize_payment_method(str(raw_payment_method))
        or raw_payment_method
    )

    # Return a readable transaction summary.
    return (
        f"ID: {transaction_id} | "
        f"{expense_name} | "
        f"₹{amount} | "
        f"{expense_date} | "
        f"{category} | "
        f"{payment_method}"
    )


# --------------------------------------------------
# 5. Deterministic financial-advice safety guard
# --------------------------------------------------

def financial_advice_safety_node(state: FinSightState):
    """Block personal financial-advice requests before any LLM is called."""

    # Retrieve the latest user message and a small amount of recent context.
    conversation_messages = state.get("messages", [])
    latest_message = get_latest_user_message(conversation_messages)

    # Context helps identify short follow-ups such as "what should I do with it?"
    recent_context = "\n".join(
        str(getattr(item, "content", ""))
        for item in conversation_messages[-6:]
    )

    # A deterministic rule-based check prevents the model from being asked
    # to answer requests for personalised financial or investment advice.
    if is_financial_advice_request(latest_message, recent_context):
        return {
            "safety_blocked": True,
            "response": FINANCIAL_ADVICE_REFUSAL,
            "messages": [AIMessage(content=FINANCIAL_ADVICE_REFUSAL)],
        }

    # Always reset the flag for a safe request in a continuing thread.
    return {"safety_blocked": False}


def route_after_safety_guard(state: FinSightState):
    """End the turn when the safety guard has returned a refusal."""
    return "end" if state.get("safety_blocked") else "supervisor"


# --------------------------------------------------
# 6. Supervisor node
# --------------------------------------------------


def normalize_payment_method(value: str) -> str | None:
    """Return the canonical label for a supported payment method.

    Matching is case-insensitive, but the stored/displayed value always uses
    the exact label offered by the FinSight payment-method selector.
    """
    if not isinstance(value, str):
        return None

    supported_methods = {
        "upi": "UPI",
        "card": "Card",
        "cash": "Cash",
        "bank transfer": "Bank transfer",
    }

    return supported_methods.get(value.strip().casefold())


def normalize_expense_name(value: str) -> str:
    """Trim an expense name and uppercase only its first character.

    The rest of the text is preserved exactly as the user entered it.
    """
    cleaned_value = value.strip()
    if not cleaned_value:
        return cleaned_value
    return cleaned_value[0].upper() + cleaned_value[1:]

def supervisor_node(state: FinSightState):
    """
    Classify the user's request and preserve conversation context.

    If the chatbot is waiting for a transaction selection, interpret
    the user's reply as a selection rather than starting a new request.
    """
    print(
        "\n[DEBUG] supervisor_node() was called",
        flush=True
    )

    # Retrieve the existing conversation history.
    messages = state.get("messages", [])

    
    # Check whether a financial operation is awaiting approval.
    if (
        state.get("approval_status") == "pending"
        and state.get("pending_operation")
    ):
        
        # Retrieve the latest user message.
        latest_message = get_latest_user_message(messages)

        
        # --------------------------------------------------
        # Reject invalid numeric months before calling Groq.
        # Supports formats such as 13/2026 and 00-2026.
        # --------------------------------------------------
        month_match = re.search(
            r"(?<!\d)(\d{1,2})[/-](\d{4})(?!\d)",
            latest_message
        )

        # Only apply this check to explicit budget creation or update requests.
        if (
            month_match
            and re.search(
                r"\b(?:create|set|update|change)\b.*\bbudget\b"
                r"|\bbudget\b.*\b(?:create|update|change)\b",
                latest_message,
                re.IGNORECASE
            )
        ):
            month_number = int(month_match.group(1))

            # Reject months outside the valid range.
            if not 1 <= month_number <= 12:
                response = (
                    "Invalid budget month. Please provide a month "
                    "between 01 and 12 in MM-YYYY format."
                )

                print(
                    "[DEBUG] Invalid budget month detected:",
                    month_match.group(0),
                    flush=True
                )

                # Stop processing without making an LLM call.
                return {
                    "intent": (
                        "edit_budget"
                        if re.search(
                            r"\b(?:update|change)\b",
                            latest_message,
                            re.IGNORECASE
                        )
                        else "create_budget"
                    ),
                    "entities": {},
                    "needs_clarification": True,
                    "response": response,
                    "messages": [AIMessage(content=response)]
                }
            
        # Classify the response without using the LLM.
        decision = classify_approval_response(latest_message)

        # Preserve the existing conversation intent and entities.
        return {
            "intent": state.get("intent"),
            "entities": state.get("entities", {}),
            "approval_decision": decision,
            "needs_clarification": decision == "unclear",
            "response": ""
        }


    # Check whether the chatbot is waiting for a transaction selection.
    if state.get("transaction_resolution_status") == "needs_selection":

        # Retrieve the latest user message.
        latest_message = get_latest_user_message(messages)

        # Extract the user's numbered selection deterministically.
        selection_number = extract_selection_number(latest_message)

        # Preserve the original request and previously extracted details.
        entities = state.get("entities", {}).copy()


        # Check whether this is a single-category spending question.
        if entities.get("analytics_type") == "category_spending":

            # Retrieve the requested category and date range.
            category_name = entities.get("category_name")
            start_date = entities.get("start_date")
            end_date = entities.get("end_date")
            period_label = entities.get("period_label")

            try:
                # Find the database ID for the requested category.
                categories = get_categories()

                matching_category = next(
                    (
                        category
                        for category in categories
                        if category["category_name"].casefold()
                        == category_name.casefold()
                    ),
                    None
                )

                if matching_category is None:
                    raise ValueError(
                        f"Category '{category_name}' was not found."
                    )

                # Retrieve spending for this category and month only.
                category_results = get_category_summary(
                    start_date=start_date,
                    end_date=end_date,
                    category_id=matching_category["category_id"]
                )

                # A missing result means no transactions were recorded.
                total_spending = (
                    category_results[0]["total_spending"]
                    if category_results else 0
                )

                # Prepare a deterministic response using database figures.
                response = (
                    f"You spent ₹{total_spending:,.2f} on "
                    f"{category_name} in {period_label}."
                )

                if total_spending == 0:
                    response += (
                        f" No {category_name} transactions were recorded "
                        f"during this month."
                    )

                return {
                    "response": response,
                    "messages": [AIMessage(content=response)],
                    "needs_clarification": False
                }

            except (ValueError, TypeError, KeyError) as error:

                # Log the error without exposing technical details.
                print(
                    "[CATEGORY ANALYTICS ERROR]",
                    str(error),
                    flush=True
                )

                response = (
                    f"I couldn't retrieve your {category_name} "
                    f"spending for {period_label}."
                )

                return {
                    "response": response,
                    "messages": [AIMessage(content=response)],
                    "needs_clarification": False
                }


        # Store the selection if one was recognised.
        if selection_number is not None:
            entities["selection_number"] = selection_number

        # Preserve the original edit/delete intent while resolving identity.
        return {
            "intent": state.get("intent"),
            "entities": entities,
            "needs_clarification": False,
            "response": ""
        }

        
    # Retrieve the latest user message before invoking the LLM.
    latest_message = get_latest_user_message(messages)

    # --------------------------------------------------
    # Resolve transaction-query follow-ups deterministically.
    # --------------------------------------------------

    # Reuse the original query filters when the user says, for example,
    # "what about last month?" after viewing this month's transactions.
    previous_intent = state.get("intent")
    previous_entities = (state.get("entities") or {}).copy()
    period_followup = re.fullmatch(
        r"\s*(?:(?:and|what about|how about)\s+)?"
        r"(this month|last month|previous month|current month)"
        r"[?.!\s]*",
        latest_message,
        re.IGNORECASE
    )

    if previous_intent == "transaction_query" and period_followup:
        requested_period = period_followup.group(1).casefold()
        previous_entities["period"] = (
            "current_month"
            if requested_period in ("this month", "current month")
            else "last_month"
        )
        previous_entities.setdefault("query_type", "list")

        print(
            "[DEBUG] Transaction period follow-up resolved:",
            previous_entities,
            flush=True
        )

        return {
            "intent": "transaction_query",
            "entities": previous_entities,
            "needs_clarification": False,
            "response": ""
        }

    # Handle highest-expense questions without asking Groq to infer a
    # recent-transactions query. Unless a period is stated, search all time.
    highest_expense_pattern = re.compile(
        r"\b(?:highest|largest|biggest|most expensive)\s+"
        r"(?:expense|expenses|transaction|purchase)\b|"
        r"\bwhat\s+(?:was|is)\s+my\s+highest\s+"
        r"(?:expense|transaction)\b",
        re.IGNORECASE
    )

    if highest_expense_pattern.search(latest_message):
        highest_entities = {
            "query_type": "highest",
            "period": "all"
        }

        # Honour an explicitly stated period in the highest-expense request.
        if re.search(r"\blast\s+month\b|\bprevious\s+month\b", latest_message, re.I):
            highest_entities["period"] = "last_month"
        elif re.search(r"\bthis\s+month\b|\bcurrent\s+month\b", latest_message, re.I):
            highest_entities["period"] = "current_month"

        return {
            "intent": "transaction_query",
            "entities": highest_entities,
            "needs_clarification": False,
            "response": ""
        }

    # --------------------------------------------------
    # Parse explicit multi-field expense edits deterministically.
    # --------------------------------------------------

    # When a transaction is already selected, accept instructions such as
    # "amount = 4700, payment method = UPI" without asking Groq to build
    # structured JSON. This avoids JSON-validation failures and supports
    # changing several fields in one message.
    if (
        state.get("intent") == "edit_expense"
        and state.get("selected_transaction")
        and not (state.get("entities") or {}).get("edit_field_pending")
    ):
        # Map user-facing field labels to the entity keys used by the graph.
        multi_edit_aliases = {
            "expense name": "expense_name",
            "name": "expense_name",
            "merchant": "expense_name",
            "amount": "amount",
            "category": "category",
            "expense date": "expense_date",
            "date": "expense_date",
            "payment method": "payment_method",
            "payment": "payment_method",
            "notes": "notes",
            "note": "notes"
        }

        # Match longer field names first so "payment method" is not
        # mistaken for the shorter alias "payment".
        field_pattern = "|".join(
            re.escape(alias)
            for alias in sorted(multi_edit_aliases, key=len, reverse=True)
        )
        assignment_pattern = re.compile(
            rf"^\s*({field_pattern})\s*[:=]\s*(.*?)\s*$",
            re.IGNORECASE
        )

        # Split only when a comma or "and" is followed by another known
        # field assignment. This preserves thousands separators in values
        # such as "amount = 4,700, payment method = UPI".
        assignment_parts = re.split(
            rf",\s*(?=(?:{field_pattern})\s*[:=])"
            rf"|\s+and\s+(?=(?:{field_pattern})\s*[:=])",
            latest_message.strip(),
            flags=re.IGNORECASE
        )

        parsed_edits = {}
        parsed_edit_fields = []
        all_parts_are_assignments = bool(assignment_parts)

        for part in assignment_parts:
            assignment_match = assignment_pattern.fullmatch(part)

            if not assignment_match:
                all_parts_are_assignments = False
                break

            field_alias = assignment_match.group(1).casefold()
            raw_value = assignment_match.group(2).strip()
            entity_key = multi_edit_aliases[field_alias]

            # Reject empty values rather than silently ignoring a field.
            if not raw_value:
                all_parts_are_assignments = False
                break

            # Convert amount text to a validated numeric value.
            if entity_key == "amount":
                parsed_amount = detect_amount_only(raw_value)

                if parsed_amount is None or parsed_amount <= 0:
                    response = (
                        "Please enter a valid amount greater than zero, "
                        "for example ₹4,700."
                    )
                    return {
                        "intent": "edit_expense",
                        "entities": (state.get("entities") or {}).copy(),
                        "needs_clarification": True,
                        "response": response,
                        "messages": [AIMessage(content=response)]
                    }

                parsed_value = parsed_amount
            elif entity_key == "expense_name":
                # Capitalize only the first character; preserve the user's
                # remaining spelling and capitalization.
                parsed_value = normalize_expense_name(raw_value)
            elif entity_key == "payment_method":
                # Accept only the payment options shown in the frontend.
                parsed_value = normalize_payment_method(raw_value)
                if parsed_value is None:
                    response = (
                        "Please choose a valid payment method: "
                        "UPI, Card, Cash, or Bank transfer."
                    )
                    return {
                        "intent": "edit_expense",
                        "entities": (state.get("entities") or {}).copy(),
                        "needs_clarification": True,
                        "response": response,
                        "messages": [AIMessage(content=response)]
                    }
            elif entity_key == "notes" and raw_value.casefold() == "clear notes":
                parsed_value = ""
            else:
                parsed_value = raw_value

            # Keep the last explicit value for a repeated field label.
            parsed_edits[entity_key] = parsed_value
            if entity_key not in parsed_edit_fields:
                parsed_edit_fields.append(entity_key)

        if all_parts_are_assignments and parsed_edits:
            updated_entities = (state.get("entities") or {}).copy()
            updated_entities.update(parsed_edits)
            updated_entities["edit_fields_submitted"] = parsed_edit_fields

            # A previous search term may also be stored as expense_name.
            # Mark the exact submitted edit fields so the preparation node
            # does not confuse search context with a requested rename.
            updated_entities.pop("edit_field_pending", None)
            updated_entities.pop("edit_field_submitted", None)

            print(
                "[DEBUG] Multi-field expense edit parsed:",
                {key: updated_entities.get(key) for key in parsed_edits},
                flush=True
            )

            return {
                "intent": "edit_expense",
                "entities": updated_entities,
                "needs_clarification": False,
                "response": ""
            }


    # --------------------------------------------------
    # Handle the two-step expense-edit clarification.
    # --------------------------------------------------

    # Map the field names the user may select to the database entity keys.
    edit_field_aliases = {
        "expense name": "expense_name",
        "name": "expense_name",
        "merchant": "expense_name",
        "amount": "amount",
        "category": "category",
        "date": "expense_date",
        "expense date": "expense_date",
        "payment": "payment_method",
        "payment method": "payment_method",
        "notes": "notes",
        "note": "notes"
    }

    # If the chatbot previously asked which field to change,
    # interpret a short reply such as "amount" deterministically.
    if (
        state.get("intent") == "edit_expense"
        and state.get("selected_transaction")
        and not (state.get("entities") or {}).get("edit_field_pending")
    ):
        selected_field = edit_field_aliases.get(latest_message.strip().casefold())

        if selected_field:
            # Remember the selected field and ask only for its new value.
            updated_entities = (state.get("entities") or {}).copy()
            updated_entities["edit_field_pending"] = selected_field

            field_prompts = {
                "expense_name": "What should the new expense name be?",
                "amount": "What should the new amount be? Please enter a positive amount, for example ₹450.",
                "category": "What should the new category be?",
                "expense_date": "What should the new date be? Please use DD-MM-YYYY.",
                "payment_method": "What should the new payment method be?",
                "notes": "What notes would you like to add? You can also reply 'clear notes'."
            }

            response = field_prompts[selected_field]

            return {
                "intent": "edit_expense",
                "entities": updated_entities,
                "needs_clarification": True,
                "response": response,
                "messages": [AIMessage(content=response)]
            }

    # If a field has already been selected, treat the latest reply as
    # its new value instead of asking the user to select the field again.
    pending_edit_field = (state.get("entities") or {}).get("edit_field_pending")

    if (
        state.get("intent") == "edit_expense"
        and state.get("selected_transaction")
        and state.get("needs_clarification")
        and pending_edit_field
    ):
        updated_entities = (state.get("entities") or {}).copy()
        raw_value = latest_message.strip()

        # Convert standalone amount replies into numeric values.
        if pending_edit_field == "amount":
            parsed_amount = detect_amount_only(raw_value)

            if parsed_amount is None or parsed_amount <= 0:
                response = (
                    "Please enter a valid amount greater than zero, "
                    "for example ₹450."
                )
                return {
                    "intent": "edit_expense",
                    "entities": updated_entities,
                    "needs_clarification": True,
                    "response": response,
                    "messages": [AIMessage(content=response)]
                }

            updated_entities["amount"] = parsed_amount

        elif pending_edit_field == "expense_name":
            # Preserve the user's text while capitalizing its first character.
            updated_entities["expense_name"] = normalize_expense_name(raw_value)

        elif pending_edit_field == "category":
            updated_entities["category"] = raw_value

        elif pending_edit_field == "expense_date":
            updated_entities["expense_date"] = raw_value

        elif pending_edit_field == "payment_method":
            # Normalize to one of the exact payment-method selector options.
            normalized_method = normalize_payment_method(raw_value)
            if normalized_method is None:
                response = (
                    "Please choose a valid payment method: "
                    "UPI, Card, Cash, or Bank transfer."
                )
                return {
                    "intent": "edit_expense",
                    "entities": updated_entities,
                    "needs_clarification": True,
                    "response": response,
                    "messages": [AIMessage(content=response)]
                }
            updated_entities["payment_method"] = normalized_method

        elif pending_edit_field == "notes":
            updated_entities["notes"] = (
                "" if raw_value.casefold() == "clear notes" else raw_value
            )

        # Preserve which field received this value so edit preparation can
        # distinguish it from older entities, such as the Swiggy search term.
        updated_entities["edit_field_submitted"] = pending_edit_field

        # Clear the pending field marker after capturing its value.
        updated_entities.pop("edit_field_pending", None)

        return {
            "intent": "edit_expense",
            "entities": updated_entities,
            "needs_clarification": False,
            "response": ""
        }


    # --------------------------------------------------
    # Ranked-list count follow-ups: change count, preserve period.
    # --------------------------------------------------
    limit_followup = resolve_analytics_limit_followup(
        latest_message,
        state.get("intent"),
        state.get("entities") or {},
    )
    if limit_followup:
        updated_context = dict(state.get("conversation_context") or {})
        updated_context["active_topic"] = limit_followup.get("analytics_type")
        updated_context["intent"] = limit_followup.get("analytics_type")
        updated_context["limit"] = limit_followup["limit"]
        print("[DEBUG] Analytics result-limit follow-up detected:",
              limit_followup, flush=True)
        return {
            "intent": "analytics",
            "entities": limit_followup,
            "conversation_context": updated_context,
            "needs_clarification": False,
            "response": "",
        }

    # --------------------------------------------------
    # General period follow-ups for all supported analytics.
    # --------------------------------------------------

    # Use the latest analytics request as the source of truth. This keeps
    # "what about last month?" attached to the active topic (top categories,
    # averages, weekly summaries, category spending, etc.) rather than
    # accidentally reclassifying it as a month-to-month comparison.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    period_followup = resolve_analytics_period_followup(
        latest_message,
        state.get("intent"),
        state.get("entities") or {},
        state.get("conversation_context") or {},
        today=datetime.now(ZoneInfo("Asia/Kolkata")).date()
    )

    if period_followup:
        period_context = {
            "active_topic": period_followup.get("analytics_type"),
            "intent": period_followup.get("analytics_type"),
            "period": {
                "start_date": period_followup["start_date"],
                "end_date": period_followup["end_date"],
                "period_label": period_followup["period_label"],
            },
        }
        # Carry category filters and ranking parameters forward where present.
        if period_followup.get("category_name"):
            period_context["category"] = period_followup["category_name"]
        if period_followup.get("category_names"):
            period_context["categories"] = period_followup["category_names"]

        print(
            "[DEBUG] General analytics period follow-up detected:",
            period_followup,
            flush=True
        )
        return {
            "intent": "analytics",
            "entities": period_followup,
            "conversation_context": period_context,
            "needs_clarification": False,
            "response": ""
        }

    # --------------------------------------------------
    # Resolve "month before that" after a monthly comparison.
    # --------------------------------------------------

    # Retrieve the previous analytics request.
    previous_intent = state.get("intent")
    previous_entities = (state.get("entities") or {}).copy()

    # Recognise contextual references to an earlier month.
    month_before_pattern = re.fullmatch(
        r"\s*(?:(?:and|what about|how about)\s+)?"
        r"(?:the\s+)?month\s+before\s+that"
        r"[?.!\s]*",
        latest_message,
        re.IGNORECASE
    )

    # Only apply this logic after an actual two-month comparison.
    if (
        month_before_pattern
        and previous_intent == "analytics"
        and previous_entities.get("current_month")
        and previous_entities.get("previous_month")
    ):
        from datetime import datetime, timedelta
        from calendar import monthrange

        try:
            # The earlier month in the previous comparison is our reference.
            reference_month = datetime.strptime(
                previous_entities["previous_month"],
                "%m-%Y"
            ).date()

            # Move back one additional calendar month.
            target_month = (
                reference_month.replace(day=1)
                - timedelta(days=1)
            )

            # Calculate the complete requested month.
            last_day = monthrange(
                target_month.year,
                target_month.month
            )[1]

            target_end = target_month.replace(day=last_day)

            # Build a single-month spending request.
            month_before_entities = {
                "analytics_type": "total_spending",
                "start_date": target_month.strftime("%d-%m-%Y"),
                "end_date": target_end.strftime("%d-%m-%Y"),
                "period_label": target_month.strftime("%B %Y")
            }

            print(
                "[DEBUG] Month-before-that follow-up resolved:",
                month_before_entities,
                flush=True
            )

            return {
                "intent": "analytics",
                "entities": month_before_entities,
                "needs_clarification": False,
                "response": ""
            }

        except (ValueError, TypeError, KeyError) as error:
            # Log failures rather than guessing the requested month.
            print(
                "[DEBUG] Could not resolve month-before-that:",
                str(error),
                flush=True
            )

    # --------------------------------------------------
    # Resolve average-daily-spending month follow-ups.
    # --------------------------------------------------

    # Retrieve the previous analytics intent and entities.
    previous_intent = state.get("intent")
    previous_entities = (state.get("entities") or {}).copy()

    # Check whether the user is asking for another month
    # after receiving an average daily spending result.
    average_daily_followup = detect_average_daily_followup(
        latest_message,
        previous_intent,
        previous_entities
    )

    # Route the follow-up directly to analytics.
    if average_daily_followup:

        print(
            "[DEBUG] Average daily spending follow-up detected:",
            average_daily_followup,
            flush=True
        )

        return {
            "intent": "analytics",
            "entities": average_daily_followup,
            "needs_clarification": False,
            "response": ""
        }


    # --------------------------------------------------
    # Resolve named-month follow-ups after spending analysis.
    # --------------------------------------------------

    # Retrieve the previous analytics request and its entities.
    previous_intent = state.get("intent")
    previous_entities = (state.get("entities") or {}).copy()

    # Resolve questions such as "What about August?"
    monthly_followup = detect_monthly_spending_followup(
        latest_message,
        previous_intent,
        previous_entities
    )

    # Route the requested month directly to analytics.
    if monthly_followup:

        print(
            "[DEBUG] Monthly spending follow-up detected:",
            monthly_followup,
            flush=True
        )

        return {
            "intent": "analytics",
            "entities": monthly_followup,
            "needs_clarification": False,
            "response": ""
        }

    # --------------------------------------------------
    # Resolve weekly-summary month follow-ups.
    # --------------------------------------------------

    # Retrieve the previous analytics intent and its entities.
    previous_intent = state.get("intent")
    previous_entities = (state.get("entities") or {}).copy()

    # Check whether the user is asking for another month
    # after receiving a weekly spending summary.
    weekly_followup = detect_weekly_summary_followup(
        latest_message,
        previous_intent,
        previous_entities
    )

    # Route the follow-up directly to analytics without Groq.
    if weekly_followup:

        print(
            "[DEBUG] Weekly-summary follow-up detected:",
            weekly_followup,
            flush=True
        )

        return {
            "intent": "analytics",
            "entities": weekly_followup,
            "needs_clarification": False,
            "response": ""
        }


    # --------------------------------------------------
    # Detect category-percentage comparisons across two months.
    # --------------------------------------------------

    # Handle comparisons before the single-month detector so a request
    # containing two months is not reduced to only its first month.
    category_comparison_request = detect_category_percentage_comparison(
        latest_message
    )

    if category_comparison_request:

        print(
            "[DEBUG] Category-percentage comparison detected:",
            category_comparison_request,
            flush=True
        )

        return {
            "intent": "analytics",
            "entities": category_comparison_request,
            "needs_clarification": False,
            "response": ""
        }


    # --------------------------------------------------
    # If the previous request was a category-ranking question and the user
    # answers with only a period (for example, "current month"), preserve
    # the requested rank instead of sending the reply to monthly comparison.
    period_only_match = re.fullmatch(
        r"\s*(?:and\s+|what\s+about\s+|how\s+about\s+)?"
        r"(this month|current month|current|this|last month|previous month|"
        r"last|previous|january|february|march|april|may|june|july|"
        r"august|september|sept|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"(?:\s+(\d{4}))?[?.!\s]*",
        latest_message,
        re.IGNORECASE
    )

    if period_only_match:
        previous_context = state.get("conversation_context") or {}
        previous_entities = state.get("entities") or {}

        # A month-only follow-up must continue the most recent analytics
        # topic, rather than searching the entire chat for an old ranking
        # question. This prevents stale category-ranking context from
        # overriding a newer overall-spending request.
        active_topic = previous_context.get("active_topic")
        previous_analytics_type = previous_entities.get("analytics_type")
        requested_period = period_only_match.group(1).casefold()
        requested_year = period_only_match.group(2)
        period_text = requested_period + (f" {requested_year}" if requested_year else "")
        requested_start, requested_end = _resolve_analytics_period(period_text)

        if active_topic == "overall_spending" or previous_analytics_type == "total_spending":
            total_followup = {
                "analytics_type": "total_spending",
                "start_date": requested_start.strftime("%d-%m-%Y"),
                "end_date": requested_end.strftime("%d-%m-%Y"),
                "period_label": requested_start.strftime("%B %Y")
            }
            return {
                "intent": "analytics",
                "entities": total_followup,
                "conversation_context": {
                    "active_topic": "overall_spending",
                    "intent": "total_spending",
                    "period": {
                        "start_date": total_followup["start_date"],
                        "end_date": total_followup["end_date"],
                        "period_label": total_followup["period_label"]
                    }
                },
                "needs_clarification": False,
                "response": ""
            }

        previous_rank_request = None

        # Preserve a category rank only when that is the active topic.
        if active_topic == "category_rank" or previous_analytics_type == "category_rank":
            for prior_message in reversed(messages[:-1]):
                if getattr(prior_message, "type", "") not in ("human", "user"):
                    continue
                prior_text = getattr(prior_message, "content", "")
                if isinstance(prior_text, str):
                    previous_rank_request = detect_category_rank_request(prior_text)
                    if previous_rank_request:
                        break

        if previous_rank_request:
            requested_period = period_only_match.group(1).casefold()
            requested_year = period_only_match.group(2)
            period_text = requested_period + (f" {requested_year}" if requested_year else "")
            requested_start, requested_end = _resolve_analytics_period(period_text)
            rank_followup = {
                **previous_rank_request,
                "start_date": requested_start.strftime("%d-%m-%Y"),
                "end_date": requested_end.strftime("%d-%m-%Y"),
                "period_label": requested_start.strftime("%B %Y")
            }

            print(
                "[DEBUG] Category-rank period follow-up detected:",
                rank_followup,
                flush=True
            )

            return {
                "intent": "analytics",
                "entities": rank_followup,
                "needs_clarification": False,
                "response": ""
            }

    # Detect category ranking and month-on-month category changes first.
    category_change_request = detect_category_change_request(latest_message)
    if category_change_request:
        print("[DEBUG] Category change analysis detected:", category_change_request, flush=True)
        return {
            "intent": "analytics",
            "entities": category_change_request,
            "needs_clarification": False,
            "response": ""
        }

    # Handle ranked category lists before single-category ranking.
    # Both use the same date resolver, so an omitted period means this month.
    top_categories_request = detect_top_categories_request(latest_message)
    if top_categories_request:
        print("[DEBUG] Top categories request detected:", top_categories_request, flush=True)
        return {
            "intent": "analytics",
            "entities": top_categories_request,
            "conversation_context": {
                "active_topic": "top_categories",
                "intent": "top_categories",
                "period": {
                    "start_date": top_categories_request["start_date"],
                    "end_date": top_categories_request["end_date"],
                    "period_label": top_categories_request["period_label"]
                }
            },
            "needs_clarification": False,
            "response": ""
        }

    category_rank_request = detect_category_rank_request(latest_message)
    weekly_top_expenses_request = (
        detect_period_analytics_request(latest_message)
        if re.search(r"\btop\s+expenses?\b", latest_message, re.IGNORECASE)
        else None
    )
    weekly_followup_request = category_rank_request or weekly_top_expenses_request
    refers_to_previous_week = bool(re.search(
        r"\b(?:that|the|this)\s+week\b", latest_message, re.IGNORECASE
    ))
    if weekly_followup_request and refers_to_previous_week:
        # Reuse the exact week from the prior weekly-highest answer. Accept
        # the numeric date format emitted by analytics_node and common
        # written-date variants, but never substitute the whole month.
        week_found = False
        for prior_message in reversed(messages[:-1]):
            if getattr(prior_message, "type", "") != "ai":
                continue
            prior_text = str(getattr(prior_message, "content", ""))
            week_match = re.search(
                r"(\d{2}-\d{2}-\d{4})\s+to\s+(\d{2}-\d{2}-\d{4})",
                prior_text,
                re.IGNORECASE
            )
            if week_match:
                from datetime import datetime
                week_start = datetime.strptime(week_match.group(1), "%d-%m-%Y").date()
                week_end = datetime.strptime(week_match.group(2), "%d-%m-%Y").date()
                weekly_followup_request.update({
                    "start_date": week_start.strftime("%d-%m-%Y"),
                    "end_date": week_end.strftime("%d-%m-%Y"),
                    "period_label": f"{week_start.strftime('%d %b')} – {week_end.strftime('%d %b %Y')}"
                })
                week_found = True
                break
        if not week_found:
            response = (
                "Which week do you mean? Please tell me the date range, "
                "or ask me to identify your highest-spending week first."
            )
            return {
                "intent": "analytics",
                "entities": {},
                "needs_clarification": True,
                "response": response,
                "messages": [AIMessage(content=response)]
            }
    if category_rank_request:
        print("[DEBUG] Category rank request detected:", category_rank_request, flush=True)
        return {
            "intent": "analytics",
            "entities": category_rank_request,
            "conversation_context": {
                "active_topic": "category_rank",
                "intent": "category_rank",
                "period": {
                    "start_date": category_rank_request["start_date"],
                    "end_date": category_rank_request["end_date"],
                    "period_label": category_rank_request["period_label"]
                }
            },
            "needs_clarification": False,
            "response": ""
        }

    if weekly_top_expenses_request and re.search(
        r"\b(?:that|the|this)\s+week\b", latest_message, re.IGNORECASE
    ):
        print("[DEBUG] Weekly top-expenses follow-up detected:", weekly_top_expenses_request, flush=True)
        return {
            "intent": "analytics",
            "entities": weekly_top_expenses_request,
            "needs_clarification": False,
            "response": ""
        }

    # Detect category-spending requests for one month.
    # --------------------------------------------------

    # Recognise category breakdown requests before calling Groq.
    category_period_request = detect_category_period_request(
        latest_message
    )

    if category_period_request:

        print(
            "[DEBUG] Category-period request detected:",
            category_period_request,
            flush=True
        )

        return {
            "intent": "analytics",
            "entities": category_period_request,
            "needs_clarification": False,
            "response": ""
        }

    # --------------------------------------------------
    # Resolve relative month comparisons deterministically.
    # --------------------------------------------------

    # Check whether the user is answering an analytics clarification.
    is_analytics_followup = (
        state.get("needs_clarification")
        and state.get("intent") == "analytics"
    )

    # Resolve phrases such as "this month with last month".
    relative_months = detect_relative_analytics_months(
        latest_message,
        is_analytics_followup=is_analytics_followup
    )

    # Extract a category explicitly mentioned in this message.
    requested_category = detect_analytics_category(latest_message)

    # Preserve the previous category only during an analytics
    # clarification, such as "this month with last month".
    if requested_category is None and is_analytics_followup:
        requested_category = (
            (state.get("entities") or {}).get("category_name")
        )

    
    # Retrieve the previous analytics entities.
    previous_entities = state.get("entities") or {}

    # Retrieve the dedicated conversation context.
    previous_context = state.get("conversation_context") or {}

    # A request for overall spending starts a new, unfiltered analytics
    # question. It must not inherit the category from an earlier answer.
    is_overall_spending_request = bool(re.search(
        r"\b(?:overall\s+spending|overall\s+expenses|"
        r"total\s+spending|total\s+spent|total\s+expenses|"
        r"show\s+(?:me\s+)?(?:my\s+)?spending|"
        r"show\s+(?:me\s+)?(?:my\s+)?expenses)\b",
        latest_message,
        re.IGNORECASE
    ))

    # Identify whether the current conversation is about category spending.
    is_category_spending_followup = (
        not is_overall_spending_request
        and (
            previous_context.get("active_topic") == "spending_analysis"
            or (
                state.get("intent") == "analytics"
                and previous_entities.get("analytics_type") == "category_spending"
            )
        )
    )

    # Reuse the previous category from conversation context first.
    previous_category = (
        (
            previous_context.get("category")
            or previous_entities.get("category_name")
        )
        if is_category_spending_followup
        else None
    )

    # Retrieve the previous date range and period label.
    previous_period = (
        previous_context.get("period")
        if is_category_spending_followup
        else None
    )

    # A follow-up such as "How much did I spend on that category?" refers
    # to the category named in the immediately preceding assistant answer.
    if not previous_category and re.search(
        r"\b(?:that|this)\s+categor(?:y|ies)\b|"
        r"\b(?:that|this|it)\b.*\bspend(?:ing)?\b",
        latest_message,
        re.IGNORECASE
    ):
        for prior_message in reversed(messages[:-1]):
            if getattr(prior_message, "type", "") == "ai":
                prior_text = str(getattr(prior_message, "content", ""))
                category_match = re.search(
                    r"\b(Food|Groceries|Shopping|Entertainment|Utilities|"
                    r"Healthcare|Education|Transport|Other)\b",
                    prior_text,
                    re.IGNORECASE
                )
                if category_match:
                    previous_category = category_match.group(1)
                    break

    # Detect a new category spending question or a natural follow-up.
    category_spending_request = detect_category_spending_request(
        latest_message,
        previous_category=previous_category,
        previous_period=previous_period
    )

        # Detect whether the user mentioned multiple expense categories.
    requested_categories = detect_analytics_categories(latest_message)

    # Check whether this is a valid spending analytics request.
    # Reuse the existing detector to resolve the requested period.
    if len(requested_categories) > 1 and category_spending_request:

        # Prepare a request containing every detected category.
        multi_category_request = {
            "analytics_type": "multi_category_spending",
            "category_names": requested_categories,
            "start_date": category_spending_request["start_date"],
            "end_date": category_spending_request["end_date"],
            "period_label": category_spending_request["period_label"]
        }

        print(
            "[DEBUG] Multi-category spending request detected:",
            multi_category_request,
            flush=True
        )

        # Preserve the requested categories and period for follow-ups.
        updated_context = {
            "active_topic": "spending_analysis",
            "intent": "multi_category_spending",
            "category": requested_categories[0],
            "categories": requested_categories,
            "period": {
                "start_date": multi_category_request["start_date"],
                "end_date": multi_category_request["end_date"],
                "period_label": multi_category_request["period_label"]
            }
        }

        # Send the request to the analytics node.
        return {
            "intent": "analytics",
            "entities": multi_category_request,
            "conversation_context": updated_context,
            "needs_clarification": False,
            "response": ""
        }

    # Route recognised requests directly to the existing analytics workflow.
    if category_spending_request:

        print(
            "[DEBUG] Category spending request detected:",
            category_spending_request,
            flush=True
        )

        # Save the resolved category and period for future follow-ups.
        updated_context = {
            "active_topic": "spending_analysis",
            "intent": "category_spending",
            "category": category_spending_request["category_name"],
            "period": {
                "start_date": category_spending_request["start_date"],
                "end_date": category_spending_request["end_date"],
                "period_label": category_spending_request["period_label"]
            }
        }

        return {
            "intent": "analytics",
            "entities": category_spending_request,
            "conversation_context": updated_context,
            "needs_clarification": False,
            "response": ""
        }


    
    # --------------------------------------------------
    # Detect generic spending summaries only after category-specific
    # requests have been checked, so "spend on Food" is not mistaken
    # for an overall total-spending question.
    period_analytics_request = detect_period_analytics_request(
        latest_message
    )
    if period_analytics_request:
        print(
            "[DEBUG] Period analytics request detected:",
            period_analytics_request,
            flush=True
        )
        period_context = None
        if period_analytics_request.get("analytics_type") == "total_spending":
            period_context = {
                "active_topic": "overall_spending",
                "intent": "total_spending",
                "period": {
                    "start_date": period_analytics_request["start_date"],
                    "end_date": period_analytics_request["end_date"],
                    "period_label": period_analytics_request["period_label"]
                }
            }

        return {
            "intent": "analytics",
            "entities": period_analytics_request,
            "conversation_context": period_context,
            "needs_clarification": False,
            "response": ""
        }

    # Detect contextual spending analysis questions.
    # --------------------------------------------------

    # Recognise both complete questions and natural follow-ups.
    spending_analysis_pattern = re.compile(
        r"\b(?:"
        r"why\s+(?:has|have|did)\s+(?:my\s+)?spending\s+"
        r"(?:increased|decreased|gone\s+up|gone\s+down|changed)|"
        r"why\s+(?:has|have|did)\s+it\s+"
        r"(?:increased|decreased|gone\s+up|gone\s+down|changed)|"
        r"why\s+is\s+it\s+(?:higher|lower|more|less)|"
        r"has\s+it\s+(?:increased|decreased|gone\s+up|gone\s+down)|"
        r"why\s+(?:am\s+i|did\s+i)\s+spend\s+(?:more|less)|"
        r"explain\s+(?:my\s+)?spending\s+(?:increase|decrease|change)|"
        r"what\s+caused\s+(?:my\s+)?spending\s+"
        r"(?:increase|decrease|change)|"
        r"what\s+caused\s+(?:the\s+)?(?:increase|decrease|change)|"
        r"is\s+that\s+(?:more|less|higher|lower)\s+than\s+last\s+month"
        r")\b",
        re.IGNORECASE
    )

    # Check whether the latest message requests a spending explanation.
    if spending_analysis_pattern.search(latest_message):

        print(
            "[DEBUG] Contextual spending analysis detected.",
            flush=True
        )

        # Retrieve the previous analytics information.
        previous_entities = state.get("entities") or {}
        previous_context = state.get("conversation_context") or {}

        # Start with any explicitly available comparison months.
        comparison_entities = {
            key: previous_entities[key]
            for key in ("current_month", "previous_month", "category_name")
            if previous_entities.get(key) is not None
        }

        # Retrieve the active spending period and category.
        period = previous_context.get("period") or {}
        category_name = (
            previous_context.get("category")
            or previous_entities.get("category_name")
        )

        # Resolve the active period into a monthly comparison.
        if (
            previous_context.get("active_topic") == "spending_analysis"
            and period.get("start_date")
        ):
            from datetime import datetime, timedelta

            try:
                # Parse the first date of the active spending period.
                active_date = datetime.strptime(
                    period["start_date"],
                    "%d-%m-%Y"
                ).date()

                # Use the active period as the current comparison month.
                current_month = active_date.strftime("%m-%Y")

                # Calculate the immediately preceding calendar month.
                previous_month_date = (
                    active_date.replace(day=1) - timedelta(days=1)
                )

                previous_month = previous_month_date.strftime("%m-%Y")

                # Prepare the exact entities expected by analytics_node.
                comparison_entities = {
                    "current_month": current_month,
                    "previous_month": previous_month
                }

                # Preserve the active category when one is available.
                if category_name:
                    comparison_entities["category_name"] = category_name

                print(
                    "[DEBUG] Contextual comparison entities:",
                    comparison_entities,
                    flush=True
                )

            except (ValueError, TypeError, KeyError) as error:

                # Keep the error visible for debugging.
                print(
                    "[DEBUG] Could not resolve the active spending period:",
                    str(error),
                    flush=True
                )

        # Route the comparison to the existing analytics workflow.
        return {
            "intent": "analytics",
            "entities": comparison_entities,
            "needs_clarification": False,
            "response": ""
        }


    # Handle a transaction ID provided after a delete clarification.
    if (
        state.get("needs_clarification")
        and state.get("intent") == "delete_expense"
    ):
        # Retrieve the latest user reply.
        latest_message = get_latest_user_message(messages).strip()

        # Accept either "23" or "transaction id 23".
        id_match = re.fullmatch(
            r"(?:transaction\s*id\s*)?#?\s*(\d+)",
            latest_message,
            re.IGNORECASE
        )

        # Continue directly when a valid numeric ID is supplied.
        if id_match:
            selected_id = int(id_match.group(1))

            return {
                "intent": "delete_expense",
                "selected_transaction_id": selected_id,
                "needs_clarification": False,
                "response": ""
            }
        

    # Recognise common questions about the chatbot's capabilities.
    capability_pattern = re.compile(
        r"\b(?:"
        r"what\s+can\s+you\s+do|"
        r"what\s+can\s+you\s+help\s+me\s+with|"
        r"how\s+can\s+you\s+help|"
        r"can\s+you\s+help\s+me\s+(?:manage\s+)?my\s+finances|"
        r"what\s+financial\s+data\s+can\s+you\s+access|"
        r"what\s+features\s+(?:do\s+you\s+have|are\s+available)|"
        r"what\s+can\s+i\s+do\s+with\s+you"
        r")\b",
        re.IGNORECASE
    )

    # Return a deterministic classification without calling Groq.
    if capability_pattern.search(latest_message):
        return {
            "intent": "general_question",
            "entities": {"response_type": "capabilities"},
            "needs_clarification": False,
            "response": ""
        }

    # Detect simple greetings and thank-you messages without calling Groq.
    greeting_pattern = re.compile(
        r"^\s*(?:"
        r"hi|hello|hey|"
        r"good\s+morning|"
        r"good\s+afternoon|"
        r"good\s+evening|"
        r"thanks|"
        r"thank\s+you|"
        r"thank\s+you\s+so\s+much"
        r")[!.?,\s]*$",
        re.IGNORECASE
    )

    # Return a deterministic response type for simple greetings.
    if greeting_pattern.fullmatch(latest_message):
        return {
            "intent": "general_question",
            "entities": {"response_type": "greeting"},
            "needs_clarification": False,
            "response": ""
        }

    
    
    # --------------------------------------------------
    # Resolve budget-status period follow-ups deterministically.
    # --------------------------------------------------

    # Preserve the budget-status intent when the user replies with a month.
    budget_period_followup = re.fullmatch(
        r"\s*(?:(?:and|what about|how about)\s+)?"
        r"(this month|current month|last month|previous month)"
        r"[?.!\s]*",
        latest_message,
        re.IGNORECASE
    )

    if state.get("intent") == "budget_query" and budget_period_followup:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        # Resolve the requested month using Indian local time.
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        target_month = today.replace(day=1)
        requested_period = budget_period_followup.group(1).casefold()

        # Move to the previous calendar month when requested.
        if requested_period in ("last month", "previous month"):
            target_month = (target_month - timedelta(days=1)).replace(day=1)

        budget_entities = (state.get("entities") or {}).copy()
        budget_entities["query_type"] = budget_entities.get("query_type", "summary")
        budget_entities["budget_month"] = target_month.strftime("%m-%Y")

        print(
            "[DEBUG] Budget period follow-up resolved:",
            budget_entities,
            flush=True
        )

        return {
            "intent": "budget_query",
            "entities": budget_entities,
            "needs_clarification": False,
            "response": ""
        }

    # --------------------------------------------------
    # Handle chatbot quick questions deterministically.
    # --------------------------------------------------

    # Total spent and top categories default to the current month.
    quick_analytics = detect_period_analytics_request(latest_message)
    if quick_analytics:
        print("[DEBUG] Quick analytics question detected:", quick_analytics, flush=True)
        return {
            "intent": "analytics",
            "entities": quick_analytics,
            "needs_clarification": False,
            "response": ""
        }

    # Recent expense is a read-only query and defaults to the latest five records.
    recent_expense_pattern = re.fullmatch(
        r"\s*(?:(?:show|view|get|display)\s+)?(?:my\s+)?"
        r"(?:recent|latest)\s+expenses?\s*[?.!]*\s*",
        latest_message,
        re.IGNORECASE
    )
    if recent_expense_pattern:
        print("[DEBUG] Recent-expense quick question detected.", flush=True)
        return {
            "intent": "transaction_query",
            "entities": {"query_type": "list", "period": "recent", "limit": 5},
            "needs_clarification": False,
            "response": ""
        }

    # Detect straightforward budget queries without calling Groq.
    budget_query_type = detect_simple_budget_query(latest_message)

    if budget_query_type:
        print(
            "[DEBUG] Deterministic budget query detected:",
            budget_query_type,
            flush=True
        )

        return {
            "intent": "budget_query",
            "entities": {
                "query_type": budget_query_type
            },
            "needs_clarification": False,
            "response": ""
        }

        # --------------------------------------------------
    # Resolve a pending transaction-search clarification.
    # --------------------------------------------------
    if (
        state.get("needs_clarification")
        and state.get("intent") == "transaction_query"
    ):

        # Treat the latest reply as the expense or merchant name.
        clarified_expense_name = latest_message.strip()

        # Retrieve the details from the original transaction request.
        previous_entities = (
            state.get("entities", {}) or {}
        ).copy()

        # Continue only when the user supplied a non-empty name.
        if clarified_expense_name:

            # Preserve existing filters and add the supplied expense name.
            previous_entities["expense_name"] = clarified_expense_name

            print(
                "[DEBUG] Transaction search clarification resolved:",
                previous_entities,
                flush=True
            )

            # Continue with the original transaction-query intent.
            return {
                "intent": "transaction_query",
                "entities": previous_entities,
                "needs_clarification": False,
                "response": ""
            }
        
    # Initialise the configured chat model.
    model = get_chat_model()

    # Use JSON mode instead of tool calling or strict JSON schema.
    structured_model = model.with_structured_output(
        IntentResult,
        method="json_mode"
    )

    # Include the system instructions and complete conversation history.
    conversation = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages
    ]

    # Avoid logging user message content because it may contain private
    # financial details. The workflow stage is sufficient for diagnostics.
    print("[DEBUG] Sending request to Groq for intent classification.", flush=True)

    # Classify the latest request using the conversation context.
    result = structured_model.invoke(conversation)

    # Extract the intent and entities returned by the LLM.
    detected_intent = result.intent
    detected_entities = dict(result.entities or {})
    
    # Retrieve the latest user message.
    latest_message = get_latest_user_message(messages)

    
    # Retrieve the previous intent and entities for clarification handling.
    previous_intent = state.get("intent")
    previous_entities = (state.get("entities") or {}).copy()

    # Handle a standalone amount when an existing budget update needs clarification.
    if (
        state.get("needs_clarification")
        and previous_intent == "edit_budget"
    ):
        # Extract replies such as 20000, ₹20,000 or 20k.
        clarified_amount = detect_amount_only(latest_message)

        if clarified_amount is not None:
            # Preserve the existing budget month and other extracted details.
            previous_entities["budget_amount"] = clarified_amount

            print(
                "[DEBUG] Budget update amount captured:",
                clarified_amount,
                flush=True
            )

            # Continue directly to the budget update preparation node.
            return {
                "intent": "edit_budget",
                "entities": previous_entities,
                "needs_clarification": False,
                "response": ""
            }
    
    # Check whether the user is responding to the existing-budget prompt.
    previous_assistant_message = next(
        (
            str(message.content)
            for message in reversed(messages[:-1])
            if isinstance(message, AIMessage)
        ),
        ""
    )

    if (
        state.get("needs_clarification")
        and previous_intent == "create_budget"
        and "Would you like to change the existing budget instead?"
        in previous_assistant_message
        and classify_approval_response(latest_message) == "approved"
    ):
        print(
            "[DEBUG] User agreed to update the existing budget.",
            flush=True
        )

        # Reuse the amount and month from the original creation request.
        return {
            "intent": "edit_budget",
            "entities": previous_entities,
            "needs_clarification": False,
            "response": ""
        }


    # Store a transaction ID only when it is resolved from
    # the actual results displayed in the previous transaction query.
    transaction_selection_update = {}

    # Check whether the user wants to edit or delete a transaction
    # from the results of the previous transaction query.
    if (
        previous_intent == "transaction_query"
        and detected_intent in ("edit_expense", "delete_expense")
    ):

        # Retrieve the exact results displayed in the previous turn.
        previous_matches = state.get("matching_transactions", [])

        # Extract an ordinal such as "first one" or "second one".
        selection_number = extract_selection_number(latest_message)

        # Resolve the ordinal only against the saved results.
        if (
            selection_number is not None
            and 1 <= selection_number <= len(previous_matches)
        ):

            # Retrieve the actual database record selected by position.
            selected_record = previous_matches[selection_number - 1]

            # Use the database ID, never an ID guessed by the LLM.
            selected_id = selected_record.get("expense_id")

            if selected_id is not None:

                transaction_selection_update = {
                    "selected_transaction_id": selected_id,
                    "transaction_resolution_status": "selected"
                }

                # Keep the resolved ID available for debugging.
                detected_entities["transaction_id"] = selected_id

                print(
                    "[DEBUG] Transaction selected from previous results:",
                    selected_id,
                    flush=True
                )

    # --------------------------------------------------
    # Preserve transaction filters when starting an edit
    # or delete request from a previous transaction search.
    # --------------------------------------------------

    # Check whether the previous request was a transaction search
    # and the current request is to edit or delete a transaction.
    if (
        previous_intent == "transaction_query"
        and detected_intent in ("edit_expense", "delete_expense")
    ):

        # Preserve the filters from the previous transaction search.
        preserved_filter_keys = [
            "expense_name",
            "start_date",
            "end_date",
            "category_id",
            "payment_method",
            "period"
        ]

        for key in preserved_filter_keys:

            # Keep newly supplied values; reuse previous values only
            # when the current request does not provide them.
            if detected_entities.get(key) is None:
                previous_value = previous_entities.get(key)

                if previous_value is not None:
                    detected_entities[key] = previous_value

        # Convert a previous month-based search into explicit dates.
        if (
            detected_entities.get("period") in (
                "current_month",
                "last_month"
            )
            and not detected_entities.get("start_date")
            and not detected_entities.get("end_date")
        ):

            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo

            # Use the Indian local date to resolve the requested month.
            today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

            # Start with the current calendar month.
            month_start = today.replace(day=1)

            # Move to the previous month when that was the original filter.
            if detected_entities["period"] == "last_month":
                month_start = month_start - timedelta(days=1)
                month_start = month_start.replace(day=1)

            # Calculate the last day of the selected month.
            if month_start.month == 12:
                next_month = month_start.replace(
                    year=month_start.year + 1,
                    month=1
                )
            else:
                next_month = month_start.replace(
                    month=month_start.month + 1
                )

            month_end = next_month - timedelta(days=1)

            # Pass the original month as explicit search boundaries.
            detected_entities["start_date"] = (
                month_start.strftime("%d-%m-%Y")
            )
            detected_entities["end_date"] = (
                month_end.strftime("%d-%m-%Y")
            )

        print(
            "[DEBUG] Preserved transaction filters for edit/delete:",
            detected_entities,
            flush=True
        )


    # --------------------------------------------------
    # Preserve expense details during clarification.
    # --------------------------------------------------
    if (
        state.get("needs_clarification")
        and state.get("intent") == "add_expense"
    ):
        # Keep the original expense intent during clarification.
        detected_intent = "add_expense"

        # Retrieve the previously extracted expense details.
        previous_expense_entities = state.get("entities", {}) or {}

        # Preserve old details while allowing newly supplied values to override them.
        detected_entities = {
            **previous_expense_entities,
            **{
                key: value
                for key, value in detected_entities.items()
                if value is not None and value != ""
            }
        }

    # --------------------------------------------------
    # Resolve and validate expense dates.
    # --------------------------------------------------
    future_expense_date = False

    if detected_intent == "add_expense":

        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        # Get today's date using the Indian timezone.
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

        # Resolve relative dates provided in the latest message.
        if re.search(r"\btoday\b", latest_message, re.IGNORECASE):
            detected_entities["expense_date"] = today.strftime("%d-%m-%Y")

        elif re.search(
            r"\bday before yesterday\b",
            latest_message,
            re.IGNORECASE
        ):
            detected_entities["expense_date"] = (
                today - timedelta(days=2)
            ).strftime("%d-%m-%Y")

        elif re.search(r"\byesterday\b", latest_message, re.IGNORECASE):
            detected_entities["expense_date"] = (
                today - timedelta(days=1)
            ).strftime("%d-%m-%Y")

        elif re.search(r"\btomorrow\b", latest_message, re.IGNORECASE):
            detected_entities["expense_date"] = (
                today + timedelta(days=1)
            ).strftime("%d-%m-%Y")

        # Check the resolved date, including dates preserved from earlier turns.
        expense_date = detected_entities.get("expense_date")

        if expense_date:
            try:
                parsed_date = datetime.strptime(
                    expense_date, "%d-%m-%Y"
                ).date()

                # Remember the validation result for the final clarification check.
                future_expense_date = parsed_date > today

            except ValueError:
                # Existing date validation handles incorrectly formatted dates.
                pass
    
    
    # --------------------------------------------------
    # Normalise the budget amount extracted by the LLM.
    # The update node expects the key "budget_amount",
    # while the LLM may return "new_budget_amount".
    # --------------------------------------------------
    if detected_intent == "edit_budget":
        if detected_entities.get("budget_amount") is None:
            detected_entities["budget_amount"] = detected_entities.get(
                "new_budget_amount"
            )

        # Keep the state consistent by using one canonical key.
        detected_entities.pop("new_budget_amount", None)

        print(
            "[DEBUG] Normalised budget update entities:",
            detected_entities,
            flush=True
        )
        
    # Check whether the user wants to reset the running scenario.
    reset_scenario = is_hypothetical_reset_request(latest_message)

    # Retrieve the previously stored hypothetical spending state.
    previous_total = float(
        state.get("hypothetical_spending_total", 0.0) or 0.0
    )
    previous_month = state.get("hypothetical_budget_month")

    # Check whether the user explicitly described another scenario.
    additional_amount = detect_hypothetical_spending(latest_message)

    # Allow a standalone amount when answering a scenario clarification.
    previous_entities = state.get("entities", {}) or {}

    if (
        additional_amount is None
        and state.get("needs_clarification")
        and state.get("intent") == "budget_query"
        and previous_entities.get("query_type") == "spending_scenario"
    ):
        additional_amount = detect_amount_only(latest_message)

    # Determine the budget month associated with this scenario.
    requested_month = (
        detected_entities.get("budget_month")
        or previous_month
    )

    # Reset the previous total when requested or when changing months.
    if (
        reset_scenario
        or (
            previous_month
            and detected_entities.get("budget_month")
            and detected_entities["budget_month"] != previous_month
        )
    ):
        previous_total = 0.0

    # Initialise the state updates with the existing scenario values.
    scenario_updates = {
        "hypothetical_spending_total": previous_total,
        "hypothetical_budget_month": requested_month
    }

    # Handle a newly detected hypothetical amount.
    if additional_amount is not None:
        detected_intent = "budget_query"
        detected_entities["query_type"] = "spending_scenario"

        # Add this amount to the running hypothetical total.
        cumulative_amount = previous_total + additional_amount

        detected_entities["additional_amount"] = additional_amount
        detected_entities["cumulative_additional_amount"] = cumulative_amount

        scenario_updates["hypothetical_spending_total"] = cumulative_amount
        scenario_updates["hypothetical_budget_month"] = requested_month

        # A valid detected amount no longer needs clarification.
        needs_clarification = False
        clarification_question = None

    elif reset_scenario:
        # Start a fresh scenario and ask for its first amount.
        detected_intent = "budget_query"
        detected_entities["query_type"] = "spending_scenario"
        detected_entities.pop("additional_amount", None)
        detected_entities.pop("cumulative_additional_amount", None)

        scenario_updates["hypothetical_spending_total"] = 0.0

        needs_clarification = True
        clarification_question = (
            "Your previous hypothetical scenario has been reset. "
            "How much additional spending would you like me to calculate?"
        )

    else:
        # Preserve the LLM's clarification result for other requests.
        needs_clarification = result.needs_clarification
        clarification_question = result.clarification_question

    # Recheck required fields after preserving and resolving expense details.
    if detected_intent == "add_expense":

        # Define all mandatory expense fields.
        required_fields = [
            "expense_name",
            "amount",
            "category",
            "expense_date",
            "payment_method"
        ]

        # Identify any fields that are still missing.
        missing_fields = [
            field
            for field in required_fields
            if detected_entities.get(field) is None
            or (
                isinstance(detected_entities.get(field), str)
                and not detected_entities[field].strip()
            )
        ]

        # Override the LLM's clarification status using actual field availability.
        needs_clarification = bool(missing_fields)

        # Clear the old clarification question when all details are available.
        if not needs_clarification:
            clarification_question = None

        # Give future-date validation priority over missing-field questions.
        if future_expense_date:
            needs_clarification = True
            clarification_question = (
                "You cannot add an expense for a future date. "
                "Please provide today or an earlier date "
                "in DD-MM-YYYY format."
            )

    # DEBUG: Inspect the final classification and scenario state.
    print("\n========== SUPERVISOR DEBUG ==========")
    print("Intent:", detected_intent, flush=True)
    print("Entities:", detected_entities, flush=True)
    print("Needs clarification:", needs_clarification, flush=True)
    print("Clarification question:", clarification_question, flush=True)
    print("======================================\n")

    # Store the final classification and updated scenario state.
    updates = {
    "intent": detected_intent,
    "entities": detected_entities,
    "needs_clarification": needs_clarification,
    "response": clarification_question or "",
    **scenario_updates,
    **transaction_selection_update
    }

    # Add a clarification question to conversation history when required.
    if clarification_question:
        updates["messages"] = [
            AIMessage(content=clarification_question)
        ]

    return updates


# --------------------------------------------------
#  Transaction Search Node
# --------------------------------------------------

def transaction_search_node(state: FinSightState):
    """
    Search for transactions matching the user's request.

    Matching transactions are stored in the graph state so they can
    be resolved in a subsequent conversation turn.
    """

    # Retrieve the entities extracted by the supervisor.
    entities = state.get("entities", {})

    # Retrieve the merchant or expense name extracted by the supervisor.
    expense_name = (
        entities.get("expense_name")
        or entities.get("merchant")
        or entities.get("merchant_name")
        or entities.get("expense")
        or entities.get("name")
    )

    # Read the category filter from the extracted entities.
    # This must be defined locally before it is used below.
    category_filter = (
        entities.get("category_name")
        or entities.get("category")
        or entities.get("category_names")
    )

    # The supervisor may return more than one category; this search
    # node currently supports one category filter at a time.
    if isinstance(category_filter, (list, tuple)):
        category_filter = category_filter[0] if category_filter else None

    # If the model mislabels a category word as an expense name, keep it
    # as a category filter instead of searching merchant names for it.
    if (
        category_filter
        and expense_name
        and str(expense_name).strip().casefold() == str(category_filter).strip().casefold()
    ):
        expense_name = None

    # Ask for clarification if there is no search term.
    if not expense_name:
        response = (
            "Which transaction would you like me to find? "
            "Please provide the merchant or expense name."
        )

        return {
            "needs_clarification": True,
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Build the search filters using only available information.
    search_filters = {
        "expense_name": expense_name,
        "start_date": entities.get("start_date"),
        "end_date": entities.get("end_date"),
        "category_id": entities.get("category_id"),
        "payment_method": entities.get("payment_method")
    }

    # Remove filters that were not provided by the user.
    search_filters = {
        key: value
        for key, value in search_filters.items()
        if value is not None
    }

    # Search the database using the existing read-only tool.
    matches = search_transactions(**search_filters)

    # Handle the case where no matching transactions were found.
    if not matches:
        response = (
            f"I couldn't find any transactions matching "
            f"'{expense_name}'. Please check the expense name."
        )

        return {
            "matching_transactions": [],
            "selected_transaction_id": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Build a compact Markdown table so the frontend can render
    # matching transactions in a consistent, easy-to-scan layout.
    transaction_rows = []

    for index, transaction in enumerate(matches, start=1):
        transaction_id = transaction.get("expense_id", "N/A")
        name = normalize_expense_name(
            str(transaction.get("expense_name", "Unknown"))
        )
        amount = float(transaction.get("amount", 0) or 0)
        expense_date = transaction.get("expense_date", "Unknown")
        category = transaction.get("category_name", "Unknown")
        raw_payment_method = transaction.get("payment_method", "Unknown")
        payment_method = (
            normalize_payment_method(str(raw_payment_method))
            or raw_payment_method
        )

        # Keep each table cell on one line and escape Markdown separators.
        def table_cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ").strip()

        transaction_rows.append(
            f"| **{index}** | {table_cell(name)} | ₹{amount:,.2f} | "
            f"{table_cell(expense_date)} | {table_cell(category)} | "
            f"{table_cell(payment_method)} | {table_cell(transaction_id)} |"
        )

    transaction_table = "\n".join([
        "| No. | Merchant / Expense | Amount | Date | Category | Payment | ID |",
        "|---:|---|---:|---|---|---|---:|",
        *transaction_rows
    ])

    # Ask the user to identify the correct transaction.
    response = (
        f"## {len(matches)} matching transaction(s)\n\n"
        "I found the following records. Please review the details and "
        "select the transaction you want to work with.\n\n"
        f"{transaction_table}\n\n"
        "**Next step:** Reply with the **number** in the first column."
    )

    # Store the candidates in their displayed order.
    return {
        "matching_transactions": matches,
        "selected_transaction_id": None,
        "transaction_resolution_status": "needs_selection",
        "response": response,
        "needs_clarification": False,
        "messages": [AIMessage(content=response)]
    }


# --------------------------------------------------
#  Transaction Details Node
# --------------------------------------------------

def transaction_details_node(state: FinSightState):
    """
    Retrieve the complete database record for the selected transaction.

    This node only reads transaction details. It does not modify
    or delete any financial records.
    """

    # Retrieve the transaction ID selected by the user.
    selected_id = state.get("selected_transaction_id")

    # Ensure a transaction ID is available before querying the database.
    if selected_id is None:
        response = (
            "No transaction has been selected. "
            "Please search for a transaction first."
        )

        return {
            "selected_transaction": None,
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the complete transaction record from the database.
    transaction = get_expense_by_id(selected_id)

    # Handle cases where the transaction no longer exists.
    if transaction is None:
        response = (
            f"I couldn't retrieve transaction ID {selected_id}. "
            "It may no longer exist in the database."
        )

        return {
            "selected_transaction": None,
            "selected_transaction_id": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Store the complete transaction record in the graph state.
    transaction = dict(transaction)

    # Display the complete retrieved record.
    response = (
        "I retrieved the complete transaction record:\n\n"
        f"{transaction}\n\n"
        "No changes have been made."
    )

    return {
        "selected_transaction": transaction,
        "response": response,
        "messages": [AIMessage(content=response)]
    }


# --------------------------------------------------
# Monthly Budget Query Node
# --------------------------------------------------

def budget_query_node(state: FinSightState):
    """
    Retrieve monthly budget information and answer specific
    budget questions using deterministic calculations.

    Supports:
    - General monthly budget summaries.
    - Budget utilisation percentages.
    - Safe daily spending guidance.
    - Hypothetical additional spending scenarios.

    This node is strictly read-only.
    """

    from datetime import datetime
    from zoneinfo import ZoneInfo
    import calendar

    # Retrieve the extracted budget query details.
    entities = state.get("entities", {})

    # Identify the requested response type.
    query_type = str(
        entities.get("query_type", "summary")
    ).lower()

    # Retrieve the requested month or default to the current month.
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    budget_month = entities.get("budget_month")

    if not budget_month:
        budget_month = today.strftime("%m-%Y")

    try:
        # Retrieve the complete monthly overview from the category-based
        # analytics layer. No legacy single-budget table is consulted.
        overview_result = get_budget_overview_tool.invoke({
            "budget_month": budget_month
        })

        # Handle validation or database retrieval failures.
        if not overview_result.get("success"):
            response = overview_result.get(
                "message",
                "Unable to retrieve the monthly budget overview."
            )
            return {
                "response": response,
                "messages": [AIMessage(content=response)]
            }

        # Use one consistent category-based result for all budget calculations.
        total_spending = float(overview_result["total_spending"])
        budget_amount = float(overview_result["total_budget"])
        category_allocations = overview_result.get("categories", [])

        # --------------------------------------------------
        # Category-specific budget status.
        # --------------------------------------------------
        category_name = entities.get("category_name")

        if query_type == "category_status" or category_name:
            # Resolve the category against the actual categories table.
            available_categories = get_categories()
            matching_category = next(
                (
                    category for category in available_categories
                    if category["category_name"].casefold()
                    == str(category_name or "").strip().casefold()
                ),
                None
            )

            # Distinguish an unknown category from a category with no
            # budget allocation for the requested month.
            if matching_category is None:
                available_names = ", ".join(
                    category["category_name"]
                    for category in available_categories
                )
                response = (
                    f"I couldn't find a category named **{category_name}** "
                    "in your FinSight categories.\n\n"
                    f"Available categories: {available_names}."
                )
                return {
                    "response": response,
                    "messages": [AIMessage(content=response)],
                    "needs_clarification": False
                }

            category_id = matching_category["category_id"]
            canonical_category_name = matching_category["category_name"]

            # Look up the category's allocation for this exact month.
            category_allocation = next(
                (
                    item for item in category_allocations
                    if int(item["category_id"]) == int(category_id)
                ),
                None
            )
            category_budget_amount = (
                float(category_allocation.get("budget_amount", 0) or 0)
                if category_allocation else 0.0
            )

            # A zero allocation means no budget was configured for this
            # category in that month; it must not be reported as ₹0 remaining.
            if category_budget_amount <= 0:
                response = (
                    f"**{canonical_category_name} Budget – {budget_month}**\n\n"
                    f"No budget has been set for **{canonical_category_name}** "
                    f"in {budget_month}. The category exists in FinSight, "
                    "but it has no budget allocation for this month."
                )
                return {
                    "response": response,
                    "messages": [AIMessage(content=response)],
                    "needs_clarification": False
                }

            # Convert the budget month into its inclusive calendar date range.
            month_date = datetime.strptime(budget_month, "%m-%Y").date()
            category_start = month_date.replace(day=1)
            category_end = month_date.replace(
                day=calendar.monthrange(month_date.year, month_date.month)[1]
            )

            # Retrieve spending only for this category and this month.
            category_spending_rows = get_category_summary(
                start_date=category_start.strftime("%d-%m-%Y"),
                end_date=category_end.strftime("%d-%m-%Y"),
                category_id=category_id
            )
            category_spending = (
                float(category_spending_rows[0]["total_spending"] or 0)
                if category_spending_rows else 0.0
            )

            category_remaining = round(
                category_budget_amount - category_spending, 2
            )
            category_utilisation = (
                category_spending / category_budget_amount * 100
            )

            response = (
                f"**{canonical_category_name} Budget – {budget_month}**\n\n"
                f"- Budget limit: ₹{category_budget_amount:,.2f}\n"
                f"- Total spending: ₹{category_spending:,.2f}\n"
                f"- Remaining budget: ₹{category_remaining:,.2f}\n"
                f"- Budget utilisation: {category_utilisation:.1f}%"
            )

            if category_remaining < 0:
                response += (
                    f"\n\nYou have exceeded your {canonical_category_name} "
                    f"budget by ₹{abs(category_remaining):,.2f}."
                )
            else:
                response += (
                    f"\n\nYou are currently within your "
                    f"{canonical_category_name} budget."
                )

            return {
                "response": response,
                "messages": [AIMessage(content=response)],
                "needs_clarification": False
            }

        # A month with all-zero category allocations has no configured budget.
        if budget_amount <= 0:
            response = (
                f"**Monthly Budget – {budget_month}**\n\n"
                "No category budgets have been set for this month.\n\n"
                f"- Total spending: ₹{total_spending:,.2f}\n"
                "- Total category budget: ₹0.00\n\n"
                "Set budgets for your categories on the Budget page to see your budget status."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)]
            }

        # Calculate remaining budget from the sum of category allocations.
        remaining_budget = round(budget_amount - total_spending, 2)

        # Calculate utilisation from the retrieved figures.
        utilisation = (
            (total_spending / budget_amount) * 100
            if budget_amount > 0
            else 0
        )

        # --------------------------------------------------
        # 1. Budget utilisation
        # --------------------------------------------------

        if query_type == "utilisation":

            response = (
                f"You have used **{utilisation:.1f}%** "
                f"of your ₹{budget_amount:,.2f} monthly budget.\n\n"
                f"- Total spending: ₹{total_spending:,.2f}\n"
                f"- Remaining budget: ₹{remaining_budget:,.2f}"
            )

        # --------------------------------------------------
        # 2. Safe daily spending guidance
        # --------------------------------------------------

        elif query_type == "daily_guidance":

            # Daily guidance is meaningful for the current month.
            requested_date = datetime.strptime(
                budget_month,
                "%m-%Y"
            ).date()

            if (
                requested_date.year != today.year
                or requested_date.month != today.month
            ):
                response = (
                    "Daily spending guidance is available for "
                    "the current month. Please ask about your "
                    "current monthly budget."
                )

            else:
                # Count the remaining calendar days, including today.
                days_in_month = calendar.monthrange(
                    today.year,
                    today.month
                )[1]

                remaining_days = days_in_month - today.day + 1

                # Calculate the daily amount available.
                if remaining_budget > 0:
                    daily_limit = remaining_budget / remaining_days
                else:
                    daily_limit = 0

                # Prepare the guidance response.
                response = (
                    f"**Your Daily Spending Guidance**\n\n"
                    f"- Monthly budget: ₹{budget_amount:,.2f}\n"
                    f"- Total spending: ₹{total_spending:,.2f}\n"
                    f"- Remaining budget: ₹{remaining_budget:,.2f}\n"
                    f"- Remaining days (including today): {remaining_days}\n\n"
                )

                if remaining_budget > 0:
                    response += (
                        f"You can spend approximately "
                        f"**₹{daily_limit:,.2f} per day** "
                        "for the rest of this month without "
                        "exceeding your budget, assuming "
                        "spending is spread evenly."
                    )
                else:
                    response += (
                        "You have already used your entire "
                        "monthly budget. Avoid additional "
                        "spending if you want to stay within it."
                    )

        # --------------------------------------------------
        # 3. Hypothetical spending scenario
        # --------------------------------------------------

        elif query_type == "spending_scenario":

            # Retrieve the additional amount extracted by the supervisor.
            raw_amount = entities.get("additional_amount")

            try:
                # Validate the hypothetical amount.
                if isinstance(raw_amount, bool) or raw_amount is None:
                    raise ValueError

                additional_amount = float(
                    str(raw_amount).replace(",", "").replace("₹", "").strip()
                )

                if not math.isfinite(additional_amount) or additional_amount < 0:
                    raise ValueError

            except (ValueError, TypeError):
                response = (
                    "How much additional spending would you "
                    "like me to calculate?"
                )

                return {
                    "needs_clarification": True,
                    "response": response,
                    "messages": [AIMessage(content=response)]
                }

                        # Retrieve the cumulative hypothetical amount from graph state.
            cumulative_amount = float(
                state.get(
                    "hypothetical_spending_total",
                    entities.get("cumulative_additional_amount", additional_amount)
                )
            )

            # Calculate projected spending using all hypothetical amounts.
            projected_spending = total_spending + cumulative_amount
            projected_remaining = budget_amount - projected_spending

            projected_utilisation = (
                (projected_spending / budget_amount) * 100
                if budget_amount > 0
                else 0
            )

            # Display both the latest amount and the cumulative scenario.
            response = (
                "**Hypothetical Spending Scenario**\n\n"
                f"- Current spending: ₹{total_spending:,.2f}\n"
                f"- Additional spending in this question: ₹{additional_amount:,.2f}\n"
                f"- Total hypothetical additional spending: ₹{cumulative_amount:,.2f}\n"
                f"- Projected spending: ₹{projected_spending:,.2f}\n"
                f"- Monthly budget: ₹{budget_amount:,.2f}\n"
                f"- Projected utilisation: {projected_utilisation:.1f}%\n"
            )

            # Explain whether the cumulative scenario exceeds the budget.
            if projected_remaining < 0:
                response += (
                    f"\nYou would exceed your monthly budget "
                    f"by **₹{abs(projected_remaining):,.2f}**."
                )
            else:
                response += (
                    f"\nYou would remain within your monthly budget "
                    f"with **₹{projected_remaining:,.2f} remaining**."
                )

            # Clearly distinguish the hypothetical calculation from actual spending.
            response += (
                "\n\nThis is only a hypothetical calculation. "
                "No transactions or budget records have been changed."
            )

        # --------------------------------------------------
        # 4. General monthly budget summary
        # --------------------------------------------------

        else:

            response = (
                f"**Monthly Budget – {budget_month}**\n\n"
                f"- Budget limit: ₹{budget_amount:,.2f}\n"
                f"- Total spending: ₹{total_spending:,.2f}\n"
                f"- Remaining budget: ₹{remaining_budget:,.2f}\n"
                f"- Budget utilisation: {utilisation:.1f}%"
            )

            # Explain the current budget status.
            if remaining_budget < 0:
                response += (
                    f"\n\nYou have exceeded your monthly budget "
                    f"by ₹{abs(remaining_budget):,.2f}."
                )
            else:
                response += (
                    "\n\nYou are currently within your monthly budget."
                )

        # Return the answer without modifying financial records.
        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    except Exception as error:

        # Log the error for debugging without exposing it to the user.
        print(
            f"Budget query error: {error}",
            flush=True
        )

        response = (
            "I couldn't retrieve your monthly budget information. "
            "Please try again."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

# --------------------------------------------------
# Transaction Query Node
# --------------------------------------------------

def transaction_query_node(state: FinSightState):
    """
    Retrieve and display expense records based on the user's request.

    Supports recent transactions, the latest transaction,
    monthly transaction counts and date-period filtering.

    This node is strictly read-only.
    """

    from datetime import datetime
    import calendar

    # Retrieve the query details extracted by the supervisor.
    entities = state.get("entities", {})

    # Read the original request so amount and named-month filters can be
    # applied deterministically even if the supervisor misses an entity.
    latest_user_text = get_latest_user_message(state.get("messages", []))

    # Running total of hypothetical spending in this conversation.
    hypothetical_spending_total: float

    # Budget month associated with the current hypothetical scenario.
    hypothetical_budget_month: str | None

    # Determine what the user wants to see.
    query_type = str(entities.get("query_type", "list")).lower()

    # Determine the requested time period.
    period = str(entities.get("period", "recent")).lower()

    # Re-apply highest-expense period instructions from the actual message.
    # This protects the result if the supervisor carries stale entities from
    # an earlier query or the model omits the requested month.
    highest_request = bool(re.search(
        r"\b(?:highest|largest|biggest|most expensive)\s+"
        r"(?:expense|expenses|transaction|purchase)\b",
        latest_user_text,
        re.IGNORECASE
    ))
    if highest_request:
        query_type = "highest"
        if re.search(r"\b(?:last|previous)\s+month\b", latest_user_text, re.IGNORECASE):
            period = "last_month"
        elif re.search(r"\b(?:this|current)\s+month\b", latest_user_text, re.IGNORECASE):
            period = "current_month"
        else:
            period = "all"

    # Resolve category names directly from the database when the model
    # misses a phrase such as "Show Food transaction".
    category_filter = (
        entities.get("category_name")
        or entities.get("category")
        or entities.get("category_names")
    )
    if isinstance(category_filter, (list, tuple)):
        category_filter = category_filter[0] if category_filter else None

    if not category_filter and re.search(
        r"\b(?:transaction|transactions|expense|expenses)\b",
        latest_user_text,
        re.IGNORECASE
    ):
        try:
            available_categories = get_categories()
            matched_categories = [
                item.get("category_name", "")
                for item in available_categories
                if item.get("category_name")
                and re.search(
                    r"(?<![A-Za-z0-9_])" + re.escape(item["category_name"]) + r"(?![A-Za-z0-9_])",
                    latest_user_text,
                    re.IGNORECASE
                )
            ]
            if matched_categories:
                category_filter = max(matched_categories, key=len)
        except Exception as category_error:
            print(
                "[DEBUG] Could not resolve category from database:",
                category_error,
                flush=True
            )

    # Retrieve a specifically requested amount from structured entities first.
    amount_filter = entities.get("amount_filter")

    # Fall back to the user's wording, for example "amount 300" or "₹300".
    if amount_filter is None:
        amount_match = re.search(
            r"\bamount\s*(?:of|is|=|:)?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)"
            r"|₹\s*([\d,]+(?:\.\d{1,2})?)",
            latest_user_text,
            re.IGNORECASE
        )
        if amount_match:
            amount_filter = next(
                (group for group in amount_match.groups() if group),
                None
            )

    # Detect a named month and year such as "November 2026" or "Nov 2026".
    month_year_filter = entities.get("month_year")
    month_names = {
        name.casefold(): number
        for number in range(1, 13)
        for name in (calendar.month_name[number], calendar.month_abbr[number])
    }
    if not month_year_filter:
        month_pattern = (
            r"\b(" + "|".join(sorted(month_names, key=len, reverse=True))
            + r")\s+(20\d{2})\b"
        )
        month_match = re.search(month_pattern, latest_user_text, re.IGNORECASE)
        if month_match:
            month_number = month_names[month_match.group(1).casefold()]
            month_year_filter = (int(month_match.group(2)), month_number)

    # Accept a structured MM-YYYY month filter as well.
    if isinstance(month_year_filter, str):
        try:
            month_number, year_number = map(int, month_year_filter.split("-"))
            month_year_filter = (year_number, month_number)
        except (ValueError, TypeError):
            month_year_filter = None

    # Keep an explicit user-supplied limit; otherwise show all records for
    # a scoped query and only five records for an unfiltered recent query.
    explicit_limit = entities.get("limit")
    has_scoped_filter = any((
        period in ("current_month", "last_month", "all")
        and period != "recent",
        category_filter,
        entities.get("category_names"),
        entities.get("expense_name"),
        entities.get("merchant"),
        entities.get("amount_filter") is not None,
        entities.get("start_date"),
        entities.get("end_date"),
        entities.get("month_year")
    ))

    if explicit_limit is None:
        limit = None if has_scoped_filter or query_type == "highest" else 5
    else:
        try:
            # Ensure an explicitly requested limit is a positive integer.
            limit = int(explicit_limit)
            if isinstance(explicit_limit, bool) or limit <= 0:
                limit = 5
        except (ValueError, TypeError):
            limit = 5

    # Retrieve all expenses using the existing read-only tool.
    try:
        expenses = get_expenses_tool.invoke({})

        # DEBUG: Inspect retrieved transactions and active query filters.
        print(
            "[DEBUG] Transaction query entities:",
            entities,
            flush=True
        )

        print(
            "[DEBUG] Total transactions retrieved:",
            len(expenses),
            flush=True
        )

        print(
            "[DEBUG] Sample expense names:",
            [
                expense.get("expense_name")
                for expense in expenses[:20]
            ],
            flush=True
        )
        
        # Ensure the tool returned a list.
        if not isinstance(expenses, list):
            raise TypeError("Expense retrieval returned an invalid result.")

    except Exception:
        response = (
            "I couldn't retrieve your transactions right now. "
            "Please try again."
        )

        # Save the exact transactions displayed to the user.
        # Their order must remain unchanged so ordinal references work.
        return {
            "matching_transactions": expenses,
            "selected_transaction_id": None,
            "selected_transaction": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Determine the current and previous calendar months.
    today = datetime.now().date()

    current_month = (today.year, today.month)

    if today.month == 1:
        previous_month = (today.year - 1, 12)
    else:
        previous_month = (today.year, today.month - 1)

    # Filter expenses when a specific month is requested.
    if period in ("current_month", "last_month"):

        target_month = (
            current_month
            if period == "current_month"
            else previous_month
        )

        filtered_expenses = []

        for expense in expenses:

            # Convert the stored DD-MM-YYYY date into a date object.
            try:
                expense_date = datetime.strptime(
                    expense["expense_date"],
                    "%d-%m-%Y"
                ).date()

            except (ValueError, TypeError, KeyError):
                # Ignore records with invalid or missing dates.
                continue

            # Keep only transactions belonging to the requested month.
            if (expense_date.year, expense_date.month) == target_month:
                filtered_expenses.append(expense)

        expenses = filtered_expenses

    # Filter by an explicitly named calendar month and year.
    if month_year_filter:
        target_year, target_month = month_year_filter
        filtered_expenses = []

        for expense in expenses:
            try:
                expense_date = datetime.strptime(
                    expense["expense_date"], "%d-%m-%Y"
                ).date()
            except (ValueError, TypeError, KeyError):
                continue

            if (expense_date.year, expense_date.month) == (
                target_year, target_month
            ):
                filtered_expenses.append(expense)

        expenses = filtered_expenses

    # Filter by an exact amount when the user supplies one.
    if amount_filter is not None:
        try:
            target_amount = float(
                str(amount_filter).replace("₹", "").replace(",", "").strip()
            )
            if not math.isfinite(target_amount) or target_amount < 0:
                raise ValueError("Invalid amount filter")
        except (ValueError, TypeError):
            response = (
                "I couldn't understand the amount you want to search for. "
                "Please provide a valid amount, for example 300."
            )
            return {
                "response": response,
                "messages": [AIMessage(content=response)]
            }

        expenses = [
            expense for expense in expenses
            if abs(float(expense.get("amount", 0) or 0) - target_amount) < 0.005
        ]

        print(
            "[DEBUG] Transaction amount filter:",
            target_amount,
            "| Matches:",
            len(expenses),
            flush=True
        )

    # --------------------------------------------------
    # Filter transactions by the supplied expense name.
    # --------------------------------------------------

    # Retrieve the expense name, if one was provided.
    expense_name = (
        entities.get("expense_name")
        or entities.get("merchant")
        or entities.get("merchant_name")
        or entities.get("expense")
        or entities.get("name")
    )

    # Apply a case-insensitive partial match.
    if expense_name:

        # Normalise the search term.
        search_term = str(expense_name).strip().casefold()

        # Keep transactions whose expense name contains the search term.
        expenses = [
            expense
            for expense in expenses
            if search_term in str(
                expense.get("expense_name", "")
            ).casefold()
        ]

        print(
            "[DEBUG] Transaction query expense-name filter:",
            search_term,
            "| Matches:",
            len(expenses),
            flush=True
        )

    # Apply a category filter when it was requested or carried forward
    # from a transaction-query follow-up.
    if category_filter:
        category_term = str(category_filter).strip().casefold()
        expenses = [
            expense for expense in expenses
            if category_term == str(expense.get("category_name") or "").strip().casefold()
        ]

        print(
            "[DEBUG] Transaction category filter:",
            category_term,
            "| Matches:",
            len(expenses),
            flush=True
        )

    # Retrieve the start and end dates extracted by the supervisor.
    start_date_value = entities.get("start_date")
    end_date_value = entities.get("end_date")

    # Apply date filtering when at least one date boundary is provided.
    if start_date_value or end_date_value:

        try:
            # Convert the requested start date into a date object.
            start_date = (
                datetime.strptime(
                    start_date_value,
                    "%d-%m-%Y"
                ).date()
                if start_date_value else None
            )

            # Convert the requested end date into a date object.
            end_date = (
                datetime.strptime(
                    end_date_value,
                    "%d-%m-%Y"
                ).date()
                if end_date_value else None
            )

            # Ensure the start date is not later than the end date.
            if start_date and end_date and start_date > end_date:
                raise ValueError("Start date cannot be after end date.")

        except ValueError:
            # Return a clear message if the requested dates are invalid.
            response = (
                "The requested date range is invalid. "
                "Please provide valid dates in DD-MM-YYYY format, "
                "with the start date before the end date."
            )

            return {
                "response": response,
                "messages": [AIMessage(content=response)]
            }

        # Store transactions that fall within the requested date range.
        filtered_expenses = []

        for expense in expenses:

            try:
                # Convert the stored transaction date into a date object.
                expense_date = datetime.strptime(
                    expense["expense_date"],
                    "%d-%m-%Y"
                ).date()

            except (ValueError, TypeError, KeyError):
                # Ignore transactions with invalid or missing dates.
                continue

            # Exclude transactions before the requested start date.
            if start_date and expense_date < start_date:
                continue

            # Exclude transactions after the requested end date.
            if end_date and expense_date > end_date:
                continue

            # Keep transactions within the requested date range.
            filtered_expenses.append(expense)

        # Update the transaction list with the filtered results.
        expenses = filtered_expenses

        print(
            "[DEBUG] Transaction date-range filter:",
            start_date_value,
            "to",
            end_date_value,
            "| Matches:",
            len(expenses),
            flush=True
        )

    # --------------------------------------------------
    # Filter transactions by the supplied expense name.
    # --------------------------------------------------


    # Return the single highest-value transaction from the filtered records.
    if query_type == "highest":
        if expenses:
            expenses = [max(
                expenses,
                key=lambda expense: float(expense.get("amount", 0) or 0)
            )]

    # Handle transaction-count requests.
    if query_type == "count":

        response = (
            f"You have made **{len(expenses)} transaction(s)** "
            "for the requested period."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Handle requests for the single latest or highest transaction.
    if query_type in ("last", "highest"):
        expenses = expenses[:1]

    elif limit is not None:
        # Apply a cap only when the request is recent or the user asked for it.
        expenses = expenses[:limit]

    # Handle cases where no matching transactions exist.
    if not expenses:

        if period == "current_month":
            period_label = "this month"
        elif period == "last_month":
            period_label = "last month"
        else:
            period_label = "the requested period"

        response = (
            f"No transactions were found for {period_label}."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Build a Markdown table using the actual database records.
    # This keeps read-only transaction results consistent with the
    # formatted transaction-search results shown elsewhere in FinSight.
    transaction_rows = []

    for index, expense in enumerate(expenses, start=1):

        # Retrieve the transaction details from the database result.
        expense_id = expense.get("expense_id", "N/A")
        name = normalize_expense_name(
            str(expense.get("expense_name", "Unknown"))
        )
        amount = float(expense.get("amount", 0))
        category = expense.get("category_name", "Unknown")
        expense_date = expense.get("expense_date", "Unknown")
        raw_payment_method = expense.get("payment_method", "Unknown")
        payment_method = (
            normalize_payment_method(str(raw_payment_method))
            or raw_payment_method
        )

        # Escape Markdown table separators and line breaks in text fields.
        def table_cell(value):
            return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()

        # Add one row per transaction, preserving the displayed order.
        transaction_rows.append(
            f"| {index} | {table_cell(name)} | "
            f"₹{amount:,.2f} | {table_cell(expense_date)} | "
            f"{table_cell(category)} | {table_cell(payment_method)} | "
            f"{table_cell(expense_id)} |"
        )

    # Add a suitable heading based on the requested transaction query.
    if query_type == "highest":
        if period == "current_month":
            heading = "Your Highest Expense This Month"
        elif period == "last_month":
            heading = "Your Highest Expense Last Month"
        else:
            heading = "Your Highest Expense"
    elif month_year_filter:
        target_year, target_month = month_year_filter
        heading = (
            f"Transactions for "
            f"{calendar.month_name[target_month]} {target_year}"
        )
    elif period == "current_month":
        heading = "Transactions This Month"
    elif period == "last_month":
        heading = "Transactions Last Month"
    elif category_filter:
        heading = f"{category_filter} Transactions"
    else:
        heading = "Your Recent Transactions"

    # Prepare a compact table that is easier to scan than separate lists.
    response = (
        f"**{heading}**\n\n"
        f"I found **{len(expenses)} transaction(s)**.\n\n"
        "| No. | Expense | Amount | Date | Category | Payment | Transaction ID |\n"
        "|---:|---|---:|---|---|---|---:|\n"
        + "\n".join(transaction_rows)
    )

    # Save the exact transactions displayed to the user.
    # This preserves their order for selections such as "first one".
    return {
        "matching_transactions": expenses,
        "selected_transaction_id": None,
        "selected_transaction": None,
        "transaction_resolution_status": "not_required",
        "response": response,
        "messages": [AIMessage(content=response)]
    }


# --------------------------------------------------
# Category Budget Preparation Nodes
# --------------------------------------------------

def _resolve_budget_category(category_name):
    """Resolve a user-provided category name against the database."""
    cleaned = str(category_name or "").strip()
    if not cleaned:
        raise ValueError("A category name is required.")

    matches = [
        item for item in get_categories()
        if str(item.get("category_name", "")).casefold() == cleaned.casefold()
    ]
    if not matches:
        available = ", ".join(
            item.get("category_name", "") for item in get_categories()
        )
        raise ValueError(
            f"'{cleaned}' is not a valid category. Available categories: {available}."
        )
    return matches[0]


def _budget_month_or_current(value):
    """Validate a supplied month or default to the current local month."""
    from datetime import datetime
    month = value or datetime.now().strftime("%m-%Y")
    validate_budget_month(month)
    return month


def _budget_clarification(message):
    return {
        "pending_operation": None,
        "approval_status": "not_required",
        "needs_clarification": True,
        "response": message,
        "messages": [AIMessage(content=message)]
    }


def create_budget_preparation_node(state: FinSightState):
    """Prepare a new category allocation; do not write before approval."""
    entities = state.get("entities", {}) or {}
    category_name = entities.get("category_name") or entities.get("category")
    raw_amount = entities.get("budget_amount")

    if not category_name:
        return _budget_clarification("Which category would you like to set a budget for?")
    if raw_amount is None:
        return _budget_clarification(f"What budget amount would you like to set for {category_name}?")

    try:
        if isinstance(raw_amount, bool):
            raise ValueError("Please provide a valid budget amount.")
        amount = float(str(raw_amount).replace("₹", "").replace(",", "").strip())
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Budget amount must be greater than zero.")
        # A new category is valid here: it will be created only after the
        # user approves the proposed budget. Do not write to SQLite during
        # preparation, because the user may still reject or cancel the proposal.
        cleaned_category_name = str(category_name).strip()
        if not cleaned_category_name:
            raise ValueError("A category name is required.")
        if len(cleaned_category_name) > 40:
            raise ValueError("Category name cannot exceed 40 characters.")

        matching_category = next(
            (
                item for item in get_categories()
                if str(item.get("category_name", "")).casefold()
                == cleaned_category_name.casefold()
            ),
            None
        )
        category = matching_category or {
            "category_id": None,
            "category_name": cleaned_category_name
        }

        month = _budget_month_or_current(entities.get("budget_month"))
        current = (
            get_category_budget(category["category_id"], month)
            if category["category_id"] is not None
            else None
        )
        current_amount = float(current.get("budget_amount", 0) or 0) if current else 0.0
        if current_amount > 0:
            return _budget_clarification(
                f"A {category['category_name']} budget already exists for {month} "
                f"(₹{current_amount:,.2f}). Would you like to edit it instead?"
            )
    except (ValueError, TypeError) as error:
        return _budget_clarification(f"I couldn't prepare this category budget: {error}")

    pending = {"operation": "create_category_budget", "proposed": {
        "category_name": category["category_name"],
        "budget_amount": amount,
        "budget_month": month
    }}
    response = (
        "**Proposed Category Budget**\n\n"
        f"- Category: {category['category_name']}\n"
        f"- Budget amount: ₹{amount:,.2f}\n"
        f"- Budget month: {month}\n\n"
        "Would you like me to create this category budget?\n\n"
        "Reply Yes to approve, No to reject, or Cancel to discard.\n"
        "No database changes have been made."
    )
    return {"pending_operation": pending, "approval_status": "pending",
            "approval_decision": "unclear", "needs_clarification": False,
            "response": response, "messages": [AIMessage(content=response)]}


def update_budget_preparation_node(state: FinSightState):
    """Prepare category allocation edits or removals for explicit approval."""
    entities = state.get("entities", {}) or {}
    category_name = entities.get("category_name") or entities.get("category")
    intent = state.get("intent")
    is_delete = intent == "delete_budget"

    if not category_name:
        return _budget_clarification("Which category budget would you like to remove?" if is_delete else "Which category budget would you like to edit?")

    try:
        category = _resolve_budget_category(category_name)
        month = _budget_month_or_current(entities.get("budget_month"))
        current = get_category_budget(category["category_id"], month)
        current_amount = float(current.get("budget_amount", 0) or 0) if current else 0.0
        if current_amount <= 0:
            return _budget_clarification(
                f"No {category['category_name']} budget is currently set for {month}. "
                "Create one first." if not is_delete else
                f"There is no {category['category_name']} budget to remove for {month}."
            )

        if is_delete:
            operation = "delete_category_budget"
            proposed = {"category_name": category["category_name"],
                        "budget_month": month,
                        "expected_current_amount": current_amount}
            response = (
                "**Proposed Category Budget Removal**\n\n"
                f"- Category: {category['category_name']}\n"
                f"- Current allocation: ₹{current_amount:,.2f}\n"
                f"- Budget month: {month}\n\n"
                "The allocation will be reset to ₹0.00; the category itself will remain.\n\n"
                "Would you like me to remove this allocation? Reply Yes to approve, No to reject, or Cancel to discard.\n"
                "No database changes have been made."
            )
        else:
            raw_amount = entities.get("budget_amount")
            if raw_amount is None:
                return _budget_clarification(f"What new budget amount would you like for {category['category_name']}?")
            if isinstance(raw_amount, bool):
                raise ValueError("Please provide a valid budget amount.")
            amount = float(str(raw_amount).replace("₹", "").replace(",", "").strip())
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError("Budget amount must be greater than zero.")
            operation = "update_category_budget"
            proposed = {"category_name": category["category_name"],
                        "budget_amount": amount, "budget_month": month,
                        "expected_current_amount": current_amount}
            response = (
                "**Proposed Category Budget Update**\n\n"
                f"- Category: {category['category_name']}\n"
                f"- Budget month: {month}\n"
                f"- Current allocation: ₹{current_amount:,.2f}\n"
                f"- New allocation: ₹{amount:,.2f}\n\n"
                "Would you like me to update this allocation? Reply Yes to approve, No to reject, or Cancel to discard.\n"
                "No database changes have been made."
            )
    except (ValueError, TypeError) as error:
        return _budget_clarification(f"I couldn't prepare this category budget operation: {error}")

    return {"pending_operation": {"operation": operation, "proposed": proposed},
            "approval_status": "pending", "approval_decision": "unclear",
            "needs_clarification": False, "response": response,
            "messages": [AIMessage(content=response)]}


# --------------------------------------------------
# Execute Category Budget Update / Removal Node
# --------------------------------------------------

def execute_budget_update_node(state: FinSightState):
    """Execute an approved category budget edit or removal."""
    pending = state.get("pending_operation")
    if state.get("approval_status") != "approved" or state.get("approval_decision") != "approved":
        response = "The category budget change was blocked because explicit approval was not confirmed."
        return {"response": response, "messages": [AIMessage(content=response)]}
    if not isinstance(pending, dict) or pending.get("operation") not in ("update_category_budget", "delete_category_budget"):
        response = "No valid category budget update or removal was found."
        return {"response": response, "messages": [AIMessage(content=response)]}
    proposed = pending.get("proposed")
    if not isinstance(proposed, dict):
        response = "The approved category budget proposal is incomplete."
        return {"pending_operation": None, "approval_status": "not_required", "response": response, "messages": [AIMessage(content=response)]}
    try:
        if pending["operation"] == "delete_category_budget":
            result = delete_category_budget_tool.invoke({
                "category_name": proposed.get("category_name"),
                "budget_month": proposed.get("budget_month"),
                "expected_current_amount": proposed.get("expected_current_amount")
            })
            if result.get("success"):
                response = (f"The {result['category_name']} budget allocation for {result['budget_month']} "
                            f"has been removed. The allocation is now ₹0.00 (previously ₹{result['previous_amount']:,.2f}).")
            else:
                response = result.get("message", "The category budget could not be removed.")
        else:
            result = update_category_budget_tool.invoke({
                "category_name": proposed.get("category_name"),
                "budget_amount": proposed.get("budget_amount"),
                "budget_month": proposed.get("budget_month"),
                "expected_current_amount": proposed.get("expected_current_amount")
            })
            if result.get("success"):
                response = (f"The {result['category_name']} budget for {result['budget_month']} was updated successfully.\n\n"
                            f"- Previous allocation: ₹{result['previous_amount']:,.2f}\n"
                            f"- New allocation: ₹{result['budget_amount']:,.2f}")
            else:
                response = result.get("message", "The category budget could not be updated.")
    except Exception:
        response = "Something went wrong while changing the category budget. Please check the budget before trying again."
    return {"pending_operation": None, "approval_status": "not_required",
            "approval_decision": "unclear", "response": response,
            "messages": [AIMessage(content=response)]}


# ------------------------------------------------
# Delete Preparation Node
# ------------------------------------------------

# Prepare an expense deletion proposal for user approval.
def delete_preparation_node(state: FinSightState):
    """
    Prepare a deletion proposal using the complete selected transaction.

    The transaction is not deleted until explicit user approval.
    """

    # Retrieve the complete selected transaction.
    transaction = state.get("selected_transaction")

    # Ensure a transaction has been selected and retrieved.
    if not isinstance(transaction, dict):
        response = (
            "I couldn't identify the transaction to delete. "
            "Please search for the transaction again."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the transaction ID.
    expense_id = transaction.get("expense_id")

    # Stop if the transaction ID is unavailable.
    if expense_id is None:
        response = (
            "The selected transaction does not have a valid ID. "
            "Please search for it again."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Prepare the exact transaction details for deletion.
    proposed = {
        "expense_id": expense_id,
        "expense_name": transaction.get("expense_name"),
        "amount": transaction.get("amount"),
        "category_name": transaction.get("category_name"),
        "expense_date": transaction.get("expense_date"),
        "payment_method": transaction.get("payment_method"),
        "notes": transaction.get("notes")
    }

    # Store the deletion operation for the approval workflow.
    pending_operation = {
        "operation": "delete_expense",
        "proposed": proposed
    }

    # Display the exact transaction that will be deleted.
    response = (
        "Please review the expense you want to delete:\n\n"
        f"- Transaction ID: {expense_id}\n"
        f"- Expense: {proposed['expense_name']}\n"
        f"- Amount: ₹{proposed['amount']:.2f}\n"
        f"- Category: {proposed['category_name']}\n"
        f"- Date: {proposed['expense_date']}\n"
        f"- Payment method: {proposed['payment_method']}\n\n"
        "Are you sure you want to delete this expense?\n\n"
        "Reply Yes to approve, No to reject, or Cancel to discard.\n"
        "No database deletion has been executed."
    )

    # Save the operation and wait for explicit approval.
    return {
        "pending_operation": pending_operation,
        "approval_status": "pending",
        "approval_decision": "unclear",
        "needs_clarification": False,
        "response": response,
        "messages": [AIMessage(content=response)]
    }

# ------------------------------------------------
# Execute Expense Deletion Node
# ------------------------------------------------

# Execute an expense deletion after explicit user approval.
def execute_expense_deletion_node(state: FinSightState):
    """
    Delete the selected expense only when the user has explicitly
    approved the exact pending deletion proposal.
    """

    # Retrieve the pending operation and approval details.
    pending_operation = state.get("pending_operation")
    approval_status = state.get("approval_status")
    approval_decision = state.get("approval_decision")

    # Block deletion unless explicit approval has been recorded.
    if (
        approval_status != "approved"
        or approval_decision != "approved"
    ):
        response = (
            "Deletion was blocked because explicit approval "
            "has not been confirmed."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Verify that the pending operation is an expense deletion.
    if (
        not isinstance(pending_operation, dict)
        or pending_operation.get("operation") != "delete_expense"
    ):
        response = "No valid expense deletion operation was found."

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the exact transaction proposed for deletion.
    proposed = pending_operation.get("proposed")

    if not isinstance(proposed, dict):
        response = "The deletion proposal is incomplete."

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the transaction ID.
    expense_id = proposed.get("expense_id")

    if expense_id is None:
        response = "The deletion proposal does not contain a valid transaction ID."

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    try:
        # Execute the deletion using the existing database function.
        deleted = delete_expense(expense_id)

        if deleted:
            response = (
                f"Expense '{proposed.get('expense_name')}' "
                f"(Transaction ID: {expense_id}) has been deleted."
            )
        else:
            response = (
                f"Transaction ID {expense_id} could not be deleted. "
                "It may have already been removed."
            )

    except Exception:
        # Avoid exposing internal database errors in the chatbot response.
        response = (
            "Something went wrong while deleting the expense. "
            "Please try again."
        )

    # Clear the completed operation so it cannot be executed again.
    return {
        "pending_operation": None,
        "approval_status": "not_required",
        "approval_decision": "unclear",
        "response": response,
        "messages": [AIMessage(content=response)]
    }


# ------------------------------------------------
# Add Expense Preparation Node
# ------------------------------------------------

def add_expense_preparation_node(state: FinSightState):
    """
    Prepare and validate a new expense before requesting approval.

    This node does not insert anything into the database.
    It only prepares the proposed expense for user review.
    """

    # Retrieve the expense details extracted by the supervisor.
    entities = state.get("entities", {})

    # Define the required fields for a new expense.
    required_fields = {
        "expense_name": "expense name",
        "amount": "amount",
        "category": "category",
        "expense_date": "date",
        "payment_method": "payment method"
    }

    # Check whether any required information is missing.
    missing_fields = [
        label
        for field, label in required_fields.items()
        if entities.get(field) is None
        or (
            isinstance(entities.get(field), str)
            and not entities[field].strip()
        )
    ]

    # Ask the user for missing information before proceeding.
    if missing_fields:
        missing_text = ", ".join(missing_fields)

        response = (
            f"I need the following information to add this expense: "
            f"{missing_text}.\n\n"
            "Please provide the missing details."
        )

        return {
            "needs_clarification": True,
            "approval_status": "not_required",
            "pending_operation": None,
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    try:
        # --------------------------------------------------
        # Validate expense name
        # --------------------------------------------------

        expense_name = entities["expense_name"]

        if not isinstance(expense_name, str) or not expense_name.strip():
            raise ValueError("Expense name must be non-empty text.")

        expense_name = expense_name.strip()

        # --------------------------------------------------
        # Validate expense amount
        # --------------------------------------------------

        raw_amount = entities["amount"]

        # Reject boolean values before numeric conversion.
        if isinstance(raw_amount, bool):
            raise ValueError("Please provide a valid expense amount.")

        # Remove currency symbols and comma separators.
        cleaned_amount = (
            str(raw_amount)
            .replace("₹", "")
            .replace(",", "")
            .strip()
        )

        amount = float(cleaned_amount)

        # Ensure the amount is finite and greater than zero.
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Expense amount must be greater than zero.")

        # --------------------------------------------------
        # Resolve expense category
        # --------------------------------------------------

        category_value = entities["category"]

        categories = get_categories()

        # Match the supplied category name against database categories.
        matching_category = next(
            (
                category for category in categories
                if category["category_name"].casefold()
                == str(category_value).strip().casefold()
            ),
            None
        )

        if matching_category is None:
            available_categories = ", ".join(
                category["category_name"]
                for category in categories
            )

            raise ValueError(
                f"'{category_value}' is not a valid category. "
                f"Available categories: {available_categories}."
            )

        category_id = matching_category["category_id"]
        category_name = matching_category["category_name"]

        # --------------------------------------------------
        # Validate expense date
        # --------------------------------------------------

        expense_date = entities["expense_date"]

        validate_actual_expense_date(expense_date)

        # --------------------------------------------------
        # Validate payment method
        # --------------------------------------------------

        payment_method = entities["payment_method"]

        if not isinstance(payment_method, str) or not payment_method.strip():
            raise ValueError("Payment method cannot be empty.")

        payment_method = payment_method.strip()

        # --------------------------------------------------
        # Validate optional notes
        # --------------------------------------------------

        notes = entities.get("notes")

        if notes is not None and not isinstance(notes, str):
            raise ValueError("Expense notes must be text.")

        notes = notes.strip() or None if notes is not None else None

    except (ValueError, TypeError) as error:
        # Stop preparation if any expense detail is invalid.
        response = (
            f"I couldn't prepare this expense: {error}\n\n"
            "Please provide the corrected information."
        )

        return {
            "needs_clarification": True,
            "approval_status": "not_required",
            "pending_operation": None,
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Prepare the complete expense without inserting it.
    proposed = {
        "expense_name": expense_name,
        "amount": amount,
        "category_id": category_id,
        "category_name": category_name,
        "expense_date": expense_date,
        "payment_method": payment_method,
        "notes": notes
    }

    # Store the proposed operation for the approval workflow.
    pending_operation = {
        "operation": "create_expense",
        "proposed": proposed
    }

    # Display the complete expense details for user review.
    response = (
        "Please review the new expense:\n\n"
        f"- Expense: {expense_name}\n"
        f"- Amount: ₹{amount:.2f}\n"
        f"- Category: {category_name}\n"
        f"- Date: {expense_date}\n"
        f"- Payment method: {payment_method}\n"
        f"- Notes: {notes or 'None'}\n\n"
        "Would you like me to add this expense?\n\n"
        "Reply Yes to approve, No to reject, or Cancel to discard.\n"
        "No database insertion has been executed."
    )

    # Store the proposal and wait for explicit user approval.
    return {
        "pending_operation": pending_operation,
        "approval_status": "pending",
        "approval_decision": "unclear",
        "needs_clarification": False,
        "response": response,
        "messages": [AIMessage(content=response)]
    }


# ------------------------------------------------
#  Edit Preparation Node
# ------------------------------------------------

def edit_preparation_node(state: FinSightState):
    """
    Prepare a complete proposed expense record using the selected
    transaction and the changes requested by the user.

    This node does not execute any database updates.
    """

    # Retrieve the complete selected transaction.
    original = state.get("selected_transaction")

    # Retrieve the changes extracted by the supervisor.
    entities = state.get("entities", {})

    # Ensure a transaction has already been selected.
    if not original:
        response = (
            "No transaction has been selected for editing. "
            "Please identify a transaction first."
        )

        return {
            "response": response,
            "pending_operation": None,
            "approval_status": "not_required",
            "messages": [AIMessage(content=response)]
        }

    # Build the complete record using only fields accepted by update_expense().
    proposed = {
        "expense_name": original["expense_name"],
        "amount": original["amount"],
        "category_id": original["category_id"],
        "expense_date": original["expense_date"],
        "payment_method": original["payment_method"],
        "notes": original.get("notes")
    }

    # Track the fields that the user actually wants to change.
    changed_fields = []

    try:
        # --------------------------------------------------
        # Update expense name
        # --------------------------------------------------

        if "expense_name" in entities:
            new_name = entities["expense_name"]

            if not isinstance(new_name, str) or not new_name.strip():
                raise ValueError("Expense name cannot be empty.")

            proposed["expense_name"] = normalize_expense_name(new_name)

        # --------------------------------------------------
        # Update amount
        # --------------------------------------------------

        if "amount" in entities:
            raw_amount = entities["amount"]

            # Reject boolean values before converting to a number.
            if isinstance(raw_amount, bool):
                raise ValueError("Please provide a valid expense amount.")

            # Remove common currency formatting.
            cleaned_amount = (
                str(raw_amount)
                .replace("₹", "")
                .replace(",", "")
                .strip()
            )

            new_amount = float(cleaned_amount)

            # Ensure the amount is valid and positive.
            if not math.isfinite(new_amount) or new_amount <= 0:
                raise ValueError("Expense amount must be greater than zero.")

            proposed["amount"] = new_amount

        # --------------------------------------------------
        # Update category
        # --------------------------------------------------

        category_value = (
            entities.get("category")
            or entities.get("category_name")
        )

        if category_value is not None:
            categories = get_categories()

            # Find the category using its name.
            matching_category = next(
                (
                    category for category in categories
                    if category["category_name"].casefold()
                    == str(category_value).strip().casefold()
                ),
                None
            )

            if matching_category is None:
                raise ValueError(
                    f"I couldn't find the category '{category_value}'."
                )

            proposed["category_id"] = matching_category["category_id"]

        # --------------------------------------------------
        # Update expense date
        # --------------------------------------------------

        if "expense_date" in entities:
            new_date = entities["expense_date"]

            # Validate the date format and reject future expense dates.
            validate_actual_expense_date(new_date)

            proposed["expense_date"] = new_date

        # --------------------------------------------------
        # Update payment method
        # --------------------------------------------------

        if "payment_method" in entities:
            new_method = entities["payment_method"]

            if not isinstance(new_method, str) or not new_method.strip():
                raise ValueError("Payment method cannot be empty.")

            normalized_method = normalize_payment_method(new_method)
            if normalized_method is None:
                raise ValueError(
                    "Please choose a valid payment method: "
                    "UPI, Card, Cash, or Bank transfer."
                )

            proposed["payment_method"] = normalized_method

        # --------------------------------------------------
        # Update notes
        # --------------------------------------------------

        if "notes" in entities and entities["notes"] is not None:
            new_notes = entities["notes"]

            if not isinstance(new_notes, str):
                raise ValueError("Notes must be text.")

            # An empty string represents clearing the existing notes.
            proposed["notes"] = new_notes.strip() or None

    except (ValueError, TypeError) as error:
        response = (
            f"I couldn't prepare the proposed changes: {error}\n\n"
            "Please provide a valid value."
        )

        return {
            "response": response,
            "pending_operation": None,
            "approval_status": "not_required",
            "messages": [AIMessage(content=response)]
        }

    # Identify which fields have actually changed.
    for field in proposed:
        original_value = original.get(field)
        proposed_value = proposed[field]

        # Treat expense names that differ only by letter case as identical.
        # This prevents a search for "protein" from being mistaken for a
        # request to rename the existing expense "Protein".
        if field == "expense_name":
            values_match = (
                str(original_value or "").strip().casefold()
                == str(proposed_value or "").strip().casefold()
            )
        else:
            values_match = proposed_value == original_value

        if not values_match:
            changed_fields.append(field)

    # Identify which editable fields the user explicitly attempted to change.
    # This lets us distinguish an unchanged value from a missing edit request.
    requested_field_keys = {
        "expense_name": ("expense_name",),
        "amount": ("amount",),
        "category": ("category", "category_name"),
        "expense_date": ("expense_date",),
        "payment_method": ("payment_method",),
        "notes": ("notes",)
    }

    # Prefer the field explicitly selected in the clarification flow. This
    # avoids mistaking an older search entity (for example, "Swiggy") for
    # a newly requested expense-name edit.
    # Support both the original single-field marker and the new list of
    # fields parsed from a multi-field instruction.
    submitted_fields = entities.get("edit_fields_submitted") or []
    submitted_field = entities.get("edit_field_submitted")

    if submitted_field in requested_field_keys and submitted_field not in submitted_fields:
        submitted_fields.append(submitted_field)

    attempted_fields = (
        [field for field in submitted_fields if field in requested_field_keys]
        if submitted_fields
        else [
            field for field, entity_keys in requested_field_keys.items()
            if any(key in entities for key in entity_keys)
            and not (
                field == "expense_name"
                and str(entities.get("expense_name", "")).strip().casefold()
                == str(original.get("expense_name", "")).strip().casefold()
            )
        ]
    )

    # If the supplied value matches the existing value, explain that clearly
    # and reset the edit-field input so the user can choose another field.
    if not changed_fields and attempted_fields:
        field_labels = {
            "expense_name": "expense name",
            "amount": "amount",
            "category": "category",
            "expense_date": "date",
            "payment_method": "payment method",
            "notes": "notes"
        }

        same_value_details = []
        for field in attempted_fields:
            if field == "category":
                current_value = next(
                    (
                        category["category_name"]
                        for category in get_categories()
                        if category["category_id"] == original["category_id"]
                    ),
                    "the current category"
                )
            elif field == "amount":
                current_value = f"₹{float(original['amount']):,.2f}"
            else:
                current_value = original.get(field) or "empty"

            same_value_details.append(
                f"The {field_labels[field]} is already {current_value}. "
                "No change is needed."
            )

        # Clear only the attempted edit values; keep transaction context intact.
        reset_entities = entities.copy()
        reset_entities.pop("edit_field_submitted", None)
        reset_entities.pop("edit_fields_submitted", None)
        for field in attempted_fields:
            for entity_key in requested_field_keys[field]:
                reset_entities.pop(entity_key, None)

        response = (
            "\n".join(same_value_details)
            + "\n\nWhat else would you like to change in this transaction?\n\n"
            "You can change the expense name, amount, category, "
            "date, payment method or notes."
        )

        return {
            "entities": reset_entities,
            "response": response,
            "pending_operation": None,
            "approval_status": "not_required",
            "needs_clarification": True,
            "messages": [AIMessage(content=response)]
        }

    # Ask the user what they want to change if no edit value was provided.
    if not changed_fields:
        response = (
            "What would you like to change in this transaction?\n\n"
            "You can change the expense name, amount, category, "
            "date, payment method or notes."
        )

        return {
            "response": response,
            "pending_operation": None,
            "approval_status": "not_required",
            "needs_clarification": True,
            "messages": [AIMessage(content=response)]
        }

    # Prepare the operation without executing it.
    pending_operation = {
        "operation": "update_expense",
        "expense_id": original["expense_id"],
        "original": {
            field: original.get(field)
            for field in proposed
        },
        "proposed": proposed,
        "changed_fields": changed_fields
    }

    # Prepare a readable comparison of the original and proposed values.
    field_labels = {
        "expense_name": "Expense name",
        "amount": "Amount",
        "category_id": "Category ID",
        "expense_date": "Expense date",
        "payment_method": "Payment method",
        "notes": "Notes"
    }

    changes_display = "\n".join(
        (
            f"- {field_labels[field]}: "
            f"{original.get(field)} → {proposed[field]}"
        )
        for field in changed_fields
    )

    # Inform the user that the proposed changes are ready for review.
    response = (
        f"Proposed changes for transaction ID "
        f"{original['expense_id']}:\n\n"
        f"{changes_display}\n\n"
        "Would you like me to apply these changes?\n\n"
        "Reply Yes to approve, No to reject, or Cancel to discard."
        "No database updates have been executed."
    )

    # Store the proposal and mark it as awaiting approval.
    cleaned_entities = entities.copy()
    cleaned_entities.pop("edit_field_submitted", None)
    cleaned_entities.pop("edit_fields_submitted", None)

    return {
        "entities": cleaned_entities,
        "pending_operation": pending_operation,
        "approval_status": "pending",
        "response": response,
        "messages": [AIMessage(content=response)]
    }

# ------------------------------------------------
#  Approval Response Node
# ------------------------------------------------

def approval_response_node(state: FinSightState):
    """
    Process the user's response to a pending financial operation.

    Records the approval decision and allows the graph to route
    approved operations to their corresponding execution nodes.
    """

    # Retrieve the classified approval decision.
    decision = state.get("approval_decision", "unclear")

    # Retrieve the operation awaiting approval.
    pending_operation = state.get("pending_operation")

    # Identify whether this is an expense creation or update.
    operation_type = (
        pending_operation.get("operation")
        if isinstance(pending_operation, dict)
        else None
    )

    # Ensure there is an operation awaiting approval.
    if not pending_operation:
        response = (
            "There is no pending financial operation to approve. "
            "Please prepare the operation again."
        )

        return {
            "approval_status": "not_required",
            "approval_decision": "unclear",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    
    # Handle an unclear response without changing the pending operation.
    if decision == "unclear":

        response = (
            "I couldn't determine whether you want to approve "
            "or reject this operation.\n\n"
            "Please reply with Yes to approve, No to reject, "
            "or Cancel to discard it."
        )

        return {
            # Keep the operation pending until a clear decision is received.
            "pending_operation": pending_operation,

            # Continue waiting for explicit user approval or rejection.
            "approval_status": "pending",

            # Preserve the unclear decision for the current response.
            "approval_decision": "unclear",

            "response": response,
            "messages": [AIMessage(content=response)]
        }


    # Record explicit approval before routing to execution.
    if decision == "approved":
        response = (
            "Your approval has been recognised. "
            "I will now attempt to execute the proposed operation."
        )

        return {
            "approval_status": "approved",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Handle rejection without executing the operation.
    if decision == "rejected":

        # Use wording appropriate to the operation type.
        if operation_type == "create_expense":
            response = (
                "The proposed expense has been rejected. "
                "No expense has been created."
            )

        elif operation_type == "delete_expense":
            response = (
                "The proposed deletion has been rejected. "
                "The expense has not been deleted."
            )

        elif operation_type in ("create_budget", "create_category_budget"):
            response = (
                "The proposed monthly budget has been rejected. "
                "No budget has been created."
            )

        elif operation_type in ("update_budget", "update_category_budget"):
            response = (
                "The proposed monthly budget update has been rejected. "
                "Your existing budget remains unchanged."
            )

        elif operation_type == "delete_category_budget":
            response = "The proposed category budget removal has been rejected. Your allocation remains unchanged."

        else:
            response = (
                "The proposed changes have been rejected. "
                "Your original transaction remains unchanged."
            )

        return {
            "approval_status": "rejected",
            "pending_operation": None,
            "approval_decision": "rejected",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Handle cancellation without executing the operation.
    if decision == "cancelled":

        # Use wording appropriate to the operation type.
        if operation_type == "create_expense":
            response = (
                "The expense creation request has been cancelled. "
                "No expense has been created."
            )

        elif operation_type in ("create_budget", "create_category_budget"):
            response = (
                "The monthly budget creation request has been cancelled. "
                "No budget has been created."
            )

        elif operation_type in ("update_budget", "update_category_budget"):
            response = (
                "The monthly budget update request has been cancelled. "
                "Your existing budget remains unchanged."
            )

        elif operation_type == "delete_category_budget":
            response = "The category budget removal has been cancelled. Your allocation remains unchanged."

        else:
            response = (
                "The pending update has been cancelled. "
                "Your original transaction remains unchanged."
            )

        return {
            "approval_status": "cancelled",
            "pending_operation": None,
            "approval_decision": "cancelled",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Safeguard against an unexpected decision value.
    response = (
        "Please confirm, reject or cancel the proposed operation."
    )

    return {
        "approval_status": "pending",
        "response": response,
        "messages": [AIMessage(content=response)]
    }


# ------------------------------------------------
# Expense Creation Execution Node
# ------------------------------------------------

def execute_expense_creation_node(state: FinSightState):
    """
    Create a new expense only after explicit user approval.

    This node executes the database insertion using the
    approved expense details stored in the pending operation.
    """

    # Verify that explicit approval has been recorded.
    if (
        state.get("approval_status") != "approved"
        or state.get("approval_decision") != "approved"
    ):
        response = (
            "The expense was not created because "
            "explicit approval could not be verified."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the approved pending operation.
    pending_operation = state.get("pending_operation")

    # Ensure a pending operation exists.
    if not isinstance(pending_operation, dict):
        response = (
            "The approved expense details could not be found. "
            "Please prepare the expense again."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Ensure this execution node handles expense creation.
    if pending_operation.get("operation") != "create_expense":
        response = (
            "The pending operation is not an expense creation. "
            "No database insertion was performed."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the exact expense details that were approved.
    proposed = pending_operation.get("proposed")

    if not isinstance(proposed, dict):
        response = (
            "The proposed expense details are incomplete. "
            "No database insertion was performed."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Verify that all required fields are available.
    required_fields = {
        "expense_name",
        "amount",
        "category_id",
        "expense_date",
        "payment_method"
    }

    if not required_fields.issubset(proposed):
        response = (
            "Some required expense details are missing. "
            "No database insertion was performed."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    try:
        # Insert the approved expense into the database.
        expense_id = add_expense(
            expense_name=proposed["expense_name"],
            amount=proposed["amount"],
            category_id=proposed["category_id"],
            expense_date=proposed["expense_date"],
            payment_method=proposed["payment_method"],
            notes=proposed.get("notes")
        )

        # Confirm successful expense creation.
        response = (
            "Your expense has been added successfully!\n\n"
            f"- Transaction ID: {expense_id}\n"
            f"- Expense: {proposed['expense_name']}\n"
            f"- Amount: ₹{proposed['amount']:.2f}\n"
            f"- Category: {proposed.get('category_name', 'Unknown')}\n"
            f"- Date: {proposed['expense_date']}\n"
            f"- Payment method: {proposed['payment_method']}\n\n"
            "The transaction has been saved in FinSight."
        )

    except ValueError as error:
        # Handle validation errors returned by the database.
        response = (
            f"The expense could not be created: {error}\n\n"
            "Please review the details and try again."
        )

    except Exception:
        # Handle unexpected database errors safely.
        response = (
            "An unexpected error occurred while creating the expense. "
            "Please check your transactions before trying again."
        )

    # Clear the completed operation so it cannot be executed again.
    return {
        "pending_operation": None,
        "approval_status": "not_required",
        "response": response,
        "messages": [AIMessage(content=response)]
    }


# --------------------------------------------------
# Category Budget Creation Execution Node
# --------------------------------------------------

def execute_budget_creation_node(state: FinSightState):
    """Create an approved category allocation without overwriting an existing one."""
    if state.get("approval_status") != "approved" or state.get("approval_decision") != "approved":
        response = "The category budget was not created because explicit approval could not be verified."
        return {"response": response, "messages": [AIMessage(content=response)]}
    pending = state.get("pending_operation")
    if not isinstance(pending, dict) or pending.get("operation") != "create_category_budget":
        response = "The pending operation is not a category budget creation. No database changes were made."
        return {"pending_operation": None, "approval_status": "not_required", "response": response, "messages": [AIMessage(content=response)]}
    proposed = pending.get("proposed")
    if not isinstance(proposed, dict) or not all(key in proposed for key in ("category_name", "budget_amount", "budget_month")):
        response = "The approved category budget details are incomplete. No database changes were made."
        return {"pending_operation": None, "approval_status": "not_required", "response": response, "messages": [AIMessage(content=response)]}
    try:
        result = create_category_budget_tool.invoke({
            "category_name": proposed["category_name"],
            "budget_amount": proposed["budget_amount"],
            "budget_month": proposed["budget_month"]
        })
        if result.get("success"):
            response = (f"Your {result['category_name']} budget for {result['budget_month']} was created successfully.\n\n"
                        f"- Allocation: ₹{result['budget_amount']:,.2f}")
        else:
            response = result.get("message", "The category budget could not be created.")
    except Exception:
        response = "An unexpected error occurred while creating the category budget. Please check your budget before retrying."
    return {"pending_operation": None, "approval_status": "not_required",
            "approval_decision": "unclear", "response": response,
            "messages": [AIMessage(content=response)]}


# Expense Update Execution Node
# ------------------------------------------------

def execute_expense_update_node(state: FinSightState):
    """
    Execute an expense update only after explicit user approval.

    The guarded database function verifies that the original
    expense has not changed since the user reviewed the proposal.
    """

    # Confirm that approval was explicitly recorded.
    if (
        state.get("approval_status") != "approved"
        or state.get("approval_decision") != "approved"
    ):
        response = (
            "The expense update was not executed because "
            "explicit approval could not be verified."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the approved operation.
    pending_operation = state.get("pending_operation")

    if not pending_operation:
        response = (
            "The approved expense update could not be found. "
            "Please prepare the changes again."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "approval_decision": "unclear",
            "matching_transactions": [],
            "selected_transaction_id": None,
            "selected_transaction": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Ensure that this node is handling an expense update.
    if pending_operation.get("operation") != "update_expense":
        response = (
            "The pending operation is not a supported expense update. "
            "No database changes were made."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "approval_decision": "unclear",
            "matching_transactions": [],
            "selected_transaction_id": None,
            "selected_transaction": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Retrieve the exact transaction and approved values.
    expense_id = pending_operation.get("expense_id")
    original = pending_operation.get("original")
    proposed = pending_operation.get("proposed")

    if expense_id is None or not isinstance(original, dict) or not isinstance(proposed, dict):
        response = (
            "The expense update details are incomplete. "
            "No database changes were made. Please prepare the changes again."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "approval_decision": "unclear",
            "matching_transactions": [],
            "selected_transaction_id": None,
            "selected_transaction": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    try:
        # Execute the update only if the original record is unchanged.
        result = update_expense_if_unchanged(
            expense_id,
            original,
            proposed
        )

        # Handle a successful database update.
        if result == "updated":
            response = (
                f"Expense transaction ID {expense_id} "
                "has been updated successfully.\n\n"
                "The approved changes have been saved."
            )

        # Handle a record modified after the proposal was prepared.
        elif result == "changed":
            response = (
                "The expense could not be updated because its details "
                "changed after you reviewed the proposed changes.\n\n"
                "No changes from this proposal were applied. "
                "Please search for the transaction again and prepare "
                "a new update."
            )

        # Handle a transaction that no longer exists.
        elif result == "not_found":
            response = (
                f"Expense transaction ID {expense_id} could not be found. "
                "It may have been deleted. No update was applied."
            )

        else:
            response = (
                "The database returned an unexpected result. "
                "Please verify the transaction before trying again."
            )

        # Clear the completed or invalidated operation.
        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "approval_decision": "unclear",
            "matching_transactions": [],
            "selected_transaction_id": None,
            "selected_transaction": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    except Exception:
        # Do not leave an approved operation available for accidental reuse.
        response = (
            "An error occurred while updating the expense. "
            "The operation could not be confirmed as successful. "
            "Please check the transaction before preparing another update."
        )

        return {
            "pending_operation": None,
            "approval_status": "not_required",
            "approval_decision": "unclear",
            "matching_transactions": [],
            "selected_transaction_id": None,
            "selected_transaction": None,
            "transaction_resolution_status": "not_required",
            "response": response,
            "messages": [AIMessage(content=response)]
        }


# -------------------------------------------------------
#  Transaction Resolution Node and Route After Supervisor
# -------------------------------------------------------

def transaction_resolution_node(state: FinSightState):
    """
    Resolve the user's selection against the stored transaction results.

    This node only identifies a transaction. It does not modify or
    delete any financial records.
    """

    # Retrieve the transactions displayed to the user.
    matches = state.get("matching_transactions", [])

    # Retrieve the extracted selection number.
    entities = state.get("entities", {})
    selection_number = entities.get("selection_number")

    # Handle an empty candidate list safely.
    if not matches:
        response = (
            "There are no pending transaction results to select from. "
            "Please start a new transaction search."
        )

        return {
            "transaction_resolution_status": "not_required",
            "selected_transaction_id": None,
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Ask the user to select an option if no number was recognised.
    if selection_number is None:
        response = (
            "Please select a transaction by replying with its "
            "number from the list above."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Use the existing deterministic selection helper.
    resolution = resolve_transaction_selection(
        matches,
        {"selection_number": selection_number}
    )

    # Retrieve the resolution status.
    status = resolution.get("status")

    # Handle invalid selections without losing the original candidates.
    if status != "identified":
        response = (
            f"Option {selection_number} is not a valid selection. "
            f"Please choose a number between 1 and {len(matches)}."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Map the validated selection to the stored transaction.
    selected_transaction = matches[selection_number - 1]

    # Retrieve the exact database transaction ID.
    selected_id = selected_transaction.get("expense_id")

    # Do not proceed if the database result has no transaction ID.
    if selected_id is None:
        response = (
            "I identified the selected result, but its transaction ID "
            "is missing. Please check the search result."
        )

        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }

    # Display the selected transaction for user verification.
    response = (
        "You selected the following transaction:\n\n"
        f"{format_transaction(selected_transaction)}\n\n"
        "The transaction has been identified successfully. "
        "No changes have been made."
    )

    # Store the selected transaction ID and update the resolution status.
    return {
        "selected_transaction_id": selected_id,
        "transaction_resolution_status": "selected",
        "response": response,
        "messages": [AIMessage(content=response)]
    }


def route_after_supervisor(state: FinSightState):
    """
    Route the request based on its intent and transaction state.

    A merchant or expense name is sufficient to start a search.
    The user does not need to identify the exact transaction first.
    """

    # Handle a pending financial operation before other workflows.
    if (
        state.get("approval_status") == "pending"
        and state.get("pending_operation")
    ):
        return "approval_response"

    # If a transaction list is awaiting selection, resolve the reply.
    if state.get("transaction_resolution_status") == "needs_selection":
        return "transaction_resolution"

    # If the chatbot has asked the user to provide a value for a selected
    # edit field, wait for the next user message instead of re-entering
    # edit_preparation_node. Without this guard, selecting a field such as
    # "category" immediately routes back to edit preparation, which sees
    # no new value and repeats the field-selection question.
    entities = state.get("entities", {}) or {}
    if (
        state.get("intent") == "edit_expense"
        and entities.get("edit_field_pending")
        and state.get("needs_clarification")
    ):
        return "end"

    # If a transaction is already selected and no field value is pending,
    # prepare its requested edits.
    if (
        state.get("intent") == "edit_expense"
        and state.get("selected_transaction")
    ):
        return "edit_preparation"

    # Retrieve the complete record when an edit or deletion
    # has been resolved to a specific transaction ID.
    if (
        state.get("intent") in ("edit_expense", "delete_expense")
        and state.get("selected_transaction_id") is not None
    ):
        return "transaction_details"

    # Retrieve the current intent and extracted information.
    intent = state.get("intent")
    entities = state.get("entities", {})

    # For edit/delete requests, check whether we have a search term.
    if intent in ("edit_expense", "delete_expense"):
        
        # Accept the possible entity names extracted by the supervisor.
        expense_name = (
            entities.get("expense_name")
            or entities.get("merchant")
            or entities.get("merchant_name")
            or entities.get("expense")
            or entities.get("name")
        )

        # Search first if a merchant or expense name is available.
        if expense_name:
            return "transaction_search"

        # Otherwise, the supervisor's clarification is necessary.
        return "end"

    # Other requests requiring clarification should wait for the user.
    if state.get("needs_clarification"):
        return "end"

    # Route read-only transaction queries to the retrieval node.
    if intent == "transaction_query":
        return "transaction_query"
    
    # Route new expense requests to the expense preparation node.
    if intent == "add_expense":
        return "add_expense_preparation"

    # Route read-only monthly budget questions.
    if intent == "budget_query":
        return "budget_query"

    # Route monthly spending comparison questions.
    if intent == "analytics":
        return "analytics"

    # Route new monthly budget creation requests.
    if intent == "create_budget":
        return "create_budget_preparation"

    # Route existing monthly budget updates.
    if intent in ("edit_budget", "delete_budget"):
        return "update_budget_preparation"
    
    # Handle greetings, general questions and unsupported intents safely.
    if intent in (
        "general_question",
        "analytics",
        "savings_analysis",
        "end_chat",
        "unknown"
    ):
        return "general_response"

    # Stop safely if an unexpected intent reaches the router.
    return "end"


def route_after_approval_response(state: FinSightState):
    """
    Continue to execution only when explicit approval is recorded
    for a valid pending financial operation.
    """

    # Retrieve the operation awaiting approval.
    pending_operation = state.get("pending_operation")

    # Require explicit approval and a valid pending operation.
    if (
        state.get("approval_status") != "approved"
        or state.get("approval_decision") != "approved"
        or not isinstance(pending_operation, dict)
    ):
        return "end"

    # Route approved operations to their respective execution nodes.
    operation = pending_operation.get("operation")

    if operation == "create_expense":
        return "execute_expense_creation"

    if operation == "update_expense":
        return "execute_expense_update"

    if operation == "delete_expense":
        return "execute_expense_deletion"

    if operation == "create_category_budget":
        return "execute_budget_creation"

    if operation in ("update_category_budget", "delete_category_budget"):
        return "execute_budget_update"

    # Stop if the operation type is unsupported.
    return "end"


def route_after_transaction_details(state: FinSightState):
    """
    Route a retrieved transaction to the appropriate preparation node.
    """

    # Check the user's original transaction intent.
    intent = state.get("intent")

    # Prepare deletion after retrieving the complete transaction.
    if intent == "delete_expense":
        return "delete_preparation"

    # Prepare editing after retrieving the complete transaction.
    if intent == "edit_expense":
        return "edit_preparation"

    # Stop if the transaction intent is unsupported.
    return "end"


def route_after_resolution(state: FinSightState):
    """
    Continue to transaction details only when a transaction
    has been successfully identified.
    """

    # Retrieve the transaction resolution status.
    if state.get("transaction_resolution_status") == "selected":
        return "transaction_details"

    # Stop if the selection was invalid or incomplete.
    return "end"


# --------------------------------------------------
#  Build the initial LangGraph workflow
# --------------------------------------------------

def build_chatbot_graph():
    """
    Build the FinSight chatbot graph with transaction management
    and expense creation workflows.
    """

    # Initialise the LangGraph state workflow.
    workflow = StateGraph(FinSightState)

    # Register the deterministic safety guard before the LLM supervisor.
    workflow.add_node("financial_advice_safety", financial_advice_safety_node)

    # Register the remaining graph nodes.
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("general_response", general_response_node)
    workflow.add_node("analytics", analytics_node)
    workflow.add_node("transaction_search", transaction_search_node)
    workflow.add_node(
        "transaction_resolution",
        transaction_resolution_node
    )

    # Register the read-only transaction retrieval node.
    workflow.add_node(
        "transaction_query",
        transaction_query_node
    )

    # Register the transaction details retrieval node.
    workflow.add_node(
        "transaction_details",
        transaction_details_node
    )

    # Register the expense edit-preparation node.
    workflow.add_node(
        "edit_preparation",
        edit_preparation_node
    )

    # Register the expense deletion-preparation node.
    workflow.add_node(
        "delete_preparation",
        delete_preparation_node
    )

     # Register the expense creation-preparation node.
    workflow.add_node(
        "add_expense_preparation",
        add_expense_preparation_node
    )

    # Register the approved expense creation execution node.
    workflow.add_node(
        "execute_expense_creation",
        execute_expense_creation_node
    )

    # Register the approved monthly budget creation node.
    workflow.add_node(
        "execute_budget_creation",
        execute_budget_creation_node
    )

    workflow.add_node(
        "approval_response",
        approval_response_node
    )

    # Register the approved expense update execution node.
    workflow.add_node(
        "execute_expense_update",
        execute_expense_update_node
    )

    # Register the approved expense deletion execution node.
    workflow.add_node(
        "execute_expense_deletion",
        execute_expense_deletion_node
    )

    # Register the read-only monthly budget query node.
    workflow.add_node(
        "budget_query",
        budget_query_node
    )    

    # Register the monthly budget creation preparation node.
    workflow.add_node(
        "create_budget_preparation",
        create_budget_preparation_node
    )

    # Update Budget Prepartion Node.
    workflow.add_node(
    "update_budget_preparation",
    update_budget_preparation_node
    )

    # Register the approved monthly budget update node.
    workflow.add_node(
        "execute_budget_update",
        execute_budget_update_node
    )

    # Run the deterministic safety check before the supervisor can call Groq.
    workflow.add_edge(START, "financial_advice_safety")
    workflow.add_conditional_edges(
        "financial_advice_safety",
        route_after_safety_guard,
        {
            "supervisor": "supervisor",
            "end": END,
        },
    )

    # Route safe requests based on their intent and current state.
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "approval_response": "approval_response",
            "transaction_query": "transaction_query",
            "transaction_search": "transaction_search",
            "transaction_resolution": "transaction_resolution",
            "transaction_details": "transaction_details",
            "edit_preparation": "edit_preparation",
            "add_expense_preparation": "add_expense_preparation",
            "budget_query": "budget_query",
            "analytics": "analytics",
            "create_budget_preparation": "create_budget_preparation",
            "update_budget_preparation": "update_budget_preparation",
            "general_response": "general_response",
            "end": END
        }
    )

    # End the turn after generating a general conversational response.
    workflow.add_edge("general_response", END)

    # End the workflow after completing the spending analysis.
    workflow.add_edge("analytics", END)

    # End the workflow after retrieving transaction information.
    workflow.add_edge("transaction_query", END)

    # End the turn after searching for matching transactions.
    workflow.add_edge("transaction_search", END)

    # Continue to retrieve complete details after successful selection.
    workflow.add_conditional_edges(
        "transaction_resolution",
        route_after_resolution,
        {
            "transaction_details": "transaction_details",
            "end": END
        }
    )


    workflow.add_conditional_edges(
    "transaction_details",
    route_after_transaction_details,
    {
        "delete_preparation": "delete_preparation",
        "edit_preparation": "edit_preparation",
        "end": END
    }
    )

    # End the workflow after preparing the proposed expense changes.
    workflow.add_edge("edit_preparation", END)

    # Route approved financial operations to their execution nodes.
    workflow.add_conditional_edges(
        "approval_response",
        route_after_approval_response,
        {
            "execute_expense_creation": "execute_expense_creation",
            "execute_expense_update": "execute_expense_update",
            "execute_expense_deletion": "execute_expense_deletion",
            "execute_budget_creation": "execute_budget_creation",
            "execute_budget_update": "execute_budget_update",
            "end": END
        }
    )

    # End the turn after preparing the deletion proposal.
    workflow.add_edge("delete_preparation", END)

    # End the workflow after attempting the approved deletion.
    workflow.add_edge("execute_expense_deletion", END)

    # End the workflow after attempting the approved expense creation.
    workflow.add_edge("execute_expense_creation", END)

    # End the workflow after attempting the approved database update.
    workflow.add_edge("execute_expense_update", END)

    # End the turn after retrieving the monthly budget information.
    workflow.add_edge("budget_query", END)

    # End the turn after preparing the budget proposal.
    workflow.add_edge("create_budget_preparation", END)

    # End the workflow after attempting the approved budget creation.
    workflow.add_edge("execute_budget_creation", END)

    # End the workflow after attempting the approved budget update.
    workflow.add_edge("execute_budget_update", END)

    # Compile the graph with persistent conversation memory.
    return workflow.compile(checkpointer=get_checkpointer())
