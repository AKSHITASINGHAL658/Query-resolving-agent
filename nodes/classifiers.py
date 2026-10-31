# Import Rich components for terminal visualization
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize Rich console
console = Console()


def classify_intent(state):
    """
    Intent Classification Node (Rules-Based Safety Check)

    Responsibilities:
    1. Identify high-risk requests requiring human approval.
    2. Reset per-turn routing state for the Supervisor Router.

    Routing itself is NOT decided here anymore -- the LLM-based
    supervisor_router picks which specialized agent(s) to call.
    The requires_approval flag set here is passed into the
    router prompt so it is aware of the sensitivity.
    """

    # ============================================================
    # STEP 1: RETRIEVE CUSTOMER QUERY
    # ============================================================

    # Get the latest user message and convert to lowercase
    # for case-insensitive matching
    query = state["messages"][-1].content.lower()

    # ============================================================
    # STEP 2: RESET PER-TURN ROUTING STATE
    # ============================================================

    # The checkpointer persists state across turns, so the
    # router's loop counter and collected responses must be
    # cleared at the start of every new customer query.
    state["routing_hops"] = 0
    state["agent_responses"] = []
    state["router_reasoning"] = None
    state["next_agent"] = ""
    state["agent_draft_response"] = ""
    state["kb_context"] = ""
    state["human_feedback"] = None
    state["query_category"] = "supervisor_router"

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
    # STEP 4: DISPLAY SAFETY CHECK VISUALIZATION
    # ============================================================

    print(
        f"\n[REQUIRES APPROVAL] {state['requires_approval']}"
    )

    route_text = Text()

    # Routing is delegated to the LLM supervisor router
    route_text.append(
        "🔀 Next Step: ",
        style="bold white"
    )
    route_text.append(
        "LLM SUPERVISOR ROUTER\n",
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

    # Show safety summary panel
    console.print(
        Panel(
            route_text,
            title="🚦 HITL Safety Check",
            border_style="magenta",
            expand=False
        )
    )

    # Return updated workflow state
    return state
