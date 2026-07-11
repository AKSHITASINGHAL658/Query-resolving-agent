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

# Import supervisor agent
from nodes.supervisor import supervisor_agent

# Import persistent memory checkpointer
from database.checkpoint import memory


def build_graph():
    """
    Build and compile the complete customer support workflow.

    Workflow Stages:
    1. Customer Profile Extraction
    2. Intent Classification
    3. Specialized Agent Processing
    4. Human-in-the-Loop Validation (if required)
    5. Supervisor Review
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

    # Final supervisor agent
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
    # STEP 4: ROUTER PASS #1
    # CLASSIFIER → SPECIALIZED AGENTS
    # ============================================================

    builder.add_conditional_edges(
        "classify_intent",
        agent_router,
        {
            "sales": "sales",
            "technical": "technical",
            "billing": "billing",
            "account": "account",
            "memory_recall": "memory"
        }
    )

    # ============================================================
    # STEP 5: ROUTER PASS #2
    # AGENTS → SUPERVISOR OR HUMAN CHECKPOINT
    # ============================================================

    # Each specialized agent can:
    # 1. Send response directly to Supervisor
    # 2. Trigger Human Approval Checkpoint

    for agent_node in [
        "sales",
        "technical",
        "billing",
        "account"
    ]:

        builder.add_conditional_edges(
            agent_node,
            agent_router,
            {
                "human_checkpoint":
                    "human_checkpoint",

                "sales":
                    "supervisor",

                "technical":
                    "supervisor",

                "billing":
                    "supervisor",

                "account":
                    "supervisor"
            }
        )

    # ============================================================
    # STEP 6: ROUTER PASS #3
    # SUPERVISOR → END OR HUMAN REVIEW
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

    # ============================================================
    # STEP 7: RE-ENTRY PATHS
    # ============================================================

    # After human approval/rejection,
    # return to supervisor for final processing

    builder.add_edge(
        "human_checkpoint",
        "supervisor"
    )

    # Memory recall responses are also
    # passed through supervisor polishing

    builder.add_edge(
        "memory",
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