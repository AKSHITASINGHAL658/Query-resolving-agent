# Import Rich components for terminal visualization
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize Rich console
console = Console()


def classify_intent(state):
    """
    Intent Classification Node

    Responsibilities:
    1. Analyze customer queries.
    2. Determine the appropriate support category.
    3. Identify high-risk requests requiring approval.
    4. Route requests to specialized agents.
    5. Update workflow state with routing decisions.
    """

    # ============================================================
    # STEP 1: RETRIEVE CUSTOMER QUERY
    # ============================================================

    # Get the latest user message and convert to lowercase
    # for case-insensitive matching
    query = state["messages"][-1].content.lower()

    # ============================================================
    # STEP 2: INITIALIZE DEFAULT STATE VALUES
    # ============================================================

    # Default routing category
    state["query_category"] = "account"

    # By default, approval is not required
    state["requires_approval"] = False

    # ============================================================
    # STEP 3: HUMAN-IN-THE-LOOP (HITL) DETECTION
    # ============================================================

    # Identify high-risk customer requests that should not
    # be automatically approved by AI agents

    if any(word in query for word in [
        "refund",
        "cancel",
        "cancellation",
        "closure",
        "close",
        "compensation",
        "compensate",
        "escalate",
        "escalation",
        "supervisor"
    ]):

        # Mark request for manual review
        state["requires_approval"] = True

    # ============================================================
    # STEP 4: QUERY CLASSIFICATION & ROUTING
    # ============================================================

    # ------------------------------------------------------------
    # MEMORY RECALL QUERIES
    # ------------------------------------------------------------
    # Example:
    # "What was my previous support issue?"

    if any(word in query for word in [
        "previous support",
        "previous issue",
        "last time",
        "my history",
        "what was my"
    ]):

        state["query_category"] = "memory_recall"

    # ------------------------------------------------------------
    # BILLING SUPPORT QUERIES
    # ------------------------------------------------------------
    # Examples:
    # - Payment issues
    # - Invoice requests
    # - Refund requests

    elif any(word in query for word in [
        "invoice",
        "bill",
        "billing",
        "payment",
        "charge",
        "receipt",
        "refund",
        "paid",
        "transaction"
    ]):

        state["query_category"] = "billing"

    # ------------------------------------------------------------
    # SALES SUPPORT QUERIES
    # ------------------------------------------------------------
    # Examples:
    # - Pricing information
    # - Subscription plans
    # - Product features

    elif any(word in query for word in [
        "price",
        "pricing",
        "plan",
        "plans",
        "subscription",
        "cost",
        "product",
        "features",
        "buy",
        "info"
    ]):

        state["query_category"] = "sales"

    # ------------------------------------------------------------
    # TECHNICAL SUPPORT QUERIES
    # ------------------------------------------------------------
    # Examples:
    # - Login failures
    # - Installation issues
    # - Application crashes

    elif any(word in query for word in [
        "crash",
        "crashes",
        "error",
        "bug",
        "install",
        "installation",
        "login",
        "failed",
        "upload",
        "configure",
        "configuration"
    ]):

        state["query_category"] = "technical"

    # ------------------------------------------------------------
    # ACCOUNT SUPPORT QUERIES
    # ------------------------------------------------------------
    # Examples:
    # - Password reset
    # - Profile updates
    # - Account activation

    elif any(word in query for word in [
        "password",
        "reset",
        "profile",
        "update",
        "activate",
        "activation",
        "deactivate",
        "account",
        "username"
    ]):

        state["query_category"] = "account"

    # ------------------------------------------------------------
    # FALLBACK ROUTE
    # ------------------------------------------------------------

    else:
        # Unknown requests are routed to Account Support
        # as a secure default option
        state["query_category"] = "account"

    # ============================================================
    # STEP 5: LOG ROUTING DECISION
    # ============================================================

    print(
        f"\n[ROUTED TO] "
        f"{state['query_category'].upper()} | "
        f"[REQUIRES APPROVAL] "
        f"{state['requires_approval']}"
    )

    # ============================================================
    # STEP 6: DISPLAY ROUTING VISUALIZATION
    # ============================================================

    route_text = Text()

    # Display classified category
    route_text.append(
        "🎯 Target Intent Category: ",
        style="bold white"
    )
    route_text.append(
        f"{state['query_category'].upper()}\n",
        style="bold cyan"
    )

    # Display routing path
    route_text.append(
        "🔀 Router Path Directed: ",
        style="bold white"
    )
    route_text.append(
        f"{state['query_category'].upper()}\n",
        style="bold yellow underline"
    )

    # Display security decision
    route_text.append(
        "🛡️ Security Verification: ",
        style="bold white"
    )

    if state["requires_approval"]:

        route_text.append(
            "REQUIRES HUMAN APPROVAL",
            style="bold red"
        )

    else:

        route_text.append(
            "AUTO-APPROVE ELIGIBLE",
            style="bold green"
        )

    # Show routing summary panel
    console.print(
        Panel(
            route_text,
            title="🚦 Graph Router Matrix",
            border_style="magenta",
            expand=False
        )
    )

    # Return updated workflow state
    return state