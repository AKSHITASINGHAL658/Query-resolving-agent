from utils.llm import llm
from knowledge_base.sales_kb import SALES_KB

def sales_agent(state):
    """Handles product information, subscription plans, and pricing details."""
    query = state["messages"][-1].content
    state["kb_context"] = SALES_KB  # Save retrieved RAG content to state 
    # 🚨 AGENT EMERGENCY OVERWRITE
    if state.get("requires_approval") and not state.get("human_feedback"):
        state["query_category"] = "human_checkpoint"
        state["agent_draft_response"] = ""  # Strictly no summarization or draft generation
        return state
    #prompt
    prompt = f"""
    You are an expert Sales Support Agent for ABC Technologies.
    Answer the customer's query using only the provided Pricing Guide context.
    
    Context:
    {state["kb_context"]}
    
    Customer Query: {query}
    """
    
    result = llm.invoke(prompt)
    
    
    return {
        "agent_draft_response": result.content,
    "query_category": state.get("query_category"), 
    "kb_context": state.get("kb_context", "")
    }