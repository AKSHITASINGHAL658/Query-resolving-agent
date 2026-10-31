from typing import TypedDict, List, Annotated, Optional
from langgraph.graph.message import add_messages

class SupportSystemState(TypedDict, total=False):

    messages: Annotated[List, add_messages]

    # Customer Metadata tracked dynamically
    customer_id: Optional[str]
    customer_name: Optional[str]

    # Classification & Routing properties
    query_category: str
    # Knowledge Base Context
    kb_context: str
    # Specialized Agent Output Draft
    agent_draft_response: Optional[str]

    # LLM Supervisor Router tracking
    next_agent: str                  # Router decision: sales/technical/billing/account/memory/FINISH
    router_reasoning: Optional[str]  # Why the router made its latest decision (debugging)
    routing_hops: int                # Passes through supervisor_router this turn (loop guard)
    agent_responses: List[dict]      # [{"agent": ..., "response": ...}] collected this turn

    human_feedback: Optional[str]
    # Human-in-the-Loop & Supervisor Approval tracking
    requires_approval: bool     # Flag set during classification if query involves risk
    approval_status: str        # 'pending', 'approved', or 'rejected'
    supervisor_feedback: Optional[str] # Review notes from the Supervisor agent
