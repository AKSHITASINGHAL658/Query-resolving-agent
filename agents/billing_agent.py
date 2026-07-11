# assignment2agentic/agents/billing_agent.py
from utils.llm import llm
from knowledge_base.billing_kb import BILLING_KB

def billing_agent(state):
    """Handles billing queries, instantly shifting query_category to human_checkpoint if high-risk."""
    state["kb_context"] = BILLING_KB  # RAG context is saved to state 
    
    # 🚨 AGENT EMERGENCY OVERWRITE
    if state.get("requires_approval") and not state.get("human_feedback"):
        state["query_category"] = "human_checkpoint"
        state["agent_draft_response"] = ""  # Strictly no summarization or draft generation
        return state
        
    # Standard path if safe or already approved
    query = state["messages"][-1].content
    prompt = f"""
    You are an expert Billing Specialist for ABC Technologies.
    Answer the customer's query using only the provided Company Policy Document context.
    
    Context:
    {state["kb_context"]}
    
    Customer Query: {query}
    """
    result = llm.invoke(prompt)
    state["agent_draft_response"] = result.content
    return state