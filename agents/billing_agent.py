# assignment2agentic/agents/billing_agent.py
from utils.llm import llm
from knowledge_base.billing_kb import BILLING_KB

def billing_agent(state):
    """Handles billing queries. High-risk requests are gated later by the supervisor's HITL checkpoint."""
    # Append (not overwrite) so multi-hop turns keep every consulted agent's context
    state["kb_context"] = (state.get("kb_context") or "") + f"\n\n[BILLING KB]:\n{BILLING_KB}"
    
    query = state["messages"][-1].content
    prompt = f"""
    You are an expert Billing Specialist for ABC Technologies.
    Answer the customer's query using only the provided Company Policy Document context.
    
    Context:
    {BILLING_KB}
    
    Customer Query: {query}
    """
    result = llm.invoke(prompt)
    state["agent_draft_response"] = result.content
    return state