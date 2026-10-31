from utils.llm import llm
from knowledge_base.sales_kb import SALES_KB

def sales_agent(state):
    """Handles product information, subscription plans, and pricing details."""
    query = state["messages"][-1].content
    # Append (not overwrite) so multi-hop turns keep every consulted agent's context
    state["kb_context"] = (state.get("kb_context") or "") + f"\n\n[SALES KB]:\n{SALES_KB}"
    #prompt
    prompt = f"""
    You are an expert Sales Support Agent for ABC Technologies.
    Answer the customer's query using only the provided Pricing Guide context.
    
    Context:
    {SALES_KB}
    
    Customer Query: {query}
    """
    
    result = llm.invoke(prompt)
    
    
    return {
        "agent_draft_response": result.content,
    "query_category": state.get("query_category"), 
    "kb_context": state.get("kb_context", "")
    }