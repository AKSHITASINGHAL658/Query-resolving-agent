# assignment2agentic/graph/workflow.py

# Import LangGraph components
from langgraph.graph import StateGraph, END

# Import workflow state schema
from graph.state import SupportSystemState

# Import routing function
from graph.router import agent_router

# Import preprocessing nodes
from nodes.extractor_profile import extractor_profile
from nodes.classifiers import classify_intent

# Import specialized support agents
from agents.sales_agent import sales_agent
from agents.tech_agent import technical_agent
from agents.billing_agent import billing_agent
from agents.account_agent import account_agent
from agents.memory_agent import memory_agent

# Import supervisor router (LLM routing brain) and
# supervisor agent (response reviewer + HITL)
from nodes.supervisor import supervisor_router, supervisor_agent

# Import persistent memory checkpointer
from database.checkpoint import memory


def build_graph():
    """
    Build and compile the complete customer support workflow.

    Workflow Stages:
    1. Customer Profile Extraction
    2. HITL Safety Check (rules-based requires_approval flag)
    3. LLM Supervisor Router <-> Specialized Agents (loop)
    4. Supervisor Review (polish)
    5. Human-in-the-Loop Validation (if required)
    6. Final Response Delivery
    """

    # ============================================================
    # STEP 1: CREATE STATE GRAPH
    # ============================================================

    # Create a graph using the shared state schema
    builder = StateGraph(SupportSystemState)

    # ============================================================
    # STEP 2: REGISTER GRAPH NODES
    # ============================================================

    # Profile extraction node
    builder.add_node(
        "extract_profile",
        extractor_profile
    )

    # Intent classification node
    builder.add_node(
        "classify_intent",
        classify_intent
    )

    # Specialized support agents
    builder.add_node("sales", sales_agent)
    builder.add_node("technical", technical_agent)
    builder.add_node("billing", billing_agent)
    builder.add_node("account", account_agent)

    # Memory recall agent
    builder.add_node("memory", memory_agent)

    # LLM supervisor router (decides which agent runs next)
    builder.add_node("supervisor_router", supervisor_router)

    # Final supervisor agent (review / polish / HITL)
    builder.add_node("supervisor", supervisor_agent)

    # Human approval checkpoint
    builder.add_node(
        "human_checkpoint",
        lambda state: state
    )

    # ============================================================
    # STEP 3: DEFINE ENTRY POINT
    # ============================================================

    # Every customer request starts here
    builder.set_entry_point("extract_profile")

    # After profile extraction,
    # move to intent classification
    builder.add_edge(
        "extract_profile",
        "classify_intent"
    )

    # ============================================================
    # STEP 4: CLASSIFIER → SUPERVISOR ROUTER (FIRST CALL)
    # ============================================================

    builder.add_edge(
        "classify_intent",
        "supervisor_router"
    )

    # ============================================================
    # STEP 5: SUPERVISOR ROUTER → AGENT OR FINISH
    # ============================================================

    # The router's LLM decision (state["next_agent"]) picks
    # the next specialist, or FINISH to go to review

    builder.add_conditional_edges(
        "supervisor_router",
        lambda state: state["next_agent"],
        {
            "sales": "sales",
            "technical": "technical",
            "billing": "billing",
            "account": "account",
            "memory": "memory",
            "FINISH": "supervisor"
        }
    )

    # ============================================================
    # STEP 6: AGENTS REPORT BACK TO SUPERVISOR ROUTER
    # ============================================================

    # Every agent returns to the router, which can re-route
    # to another specialist after seeing the response

    for agent_node in [
        "sales",
        "technical",
        "billing",
        "account",
        "memory"
    ]:

        builder.add_edge(
            agent_node,
            "supervisor_router"
        )

    # ============================================================
    # STEP 7: SUPERVISOR REVIEW → END OR HUMAN CHECKPOINT
    # ============================================================

    builder.add_conditional_edges(
        "supervisor",
        agent_router,
        {
            "human_checkpoint":
                "human_checkpoint",

            "end":
                END
        }
    )

    # After human approval/rejection,
    # return to supervisor for final processing

    builder.add_edge(
        "human_checkpoint",
        "supervisor"
    )

    # ============================================================
    # STEP 8: COMPILE GRAPH
    # ============================================================

    return builder.compile(

        # Enable persistent conversation memory
        checkpointer=memory,

        # Pause execution before reaching
        # human checkpoint node
        interrupt_before=[
            "human_checkpoint"
        ]
    )