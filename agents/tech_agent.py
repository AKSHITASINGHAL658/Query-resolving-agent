from utils.llm import llm
from knowledge_base.tech_kb import TECH_KB

def technical_agent(state):
    """Handles application errors, installation issues, and login bugs."""
    query = state["messages"][-1].content
    # Append (not overwrite) so multi-hop turns keep every consulted agent's context
    state["kb_context"] = (state.get("kb_context") or "") + f"\n\n[TECHNICAL KB]:\n{TECH_KB}"
    #prompt
    prompt = f"""
    You are an expert Technical Support Engineer for ABC Technologies.
    Answer the customer's query using only the provided Technical Manual context.
    
    Context:
    {TECH_KB}
    
    Customer Query: {query}
    """
    
    result = llm.invoke(prompt)
    
    return {
        "agent_draft_response": result.content,
    "query_category": state.get("query_category"), 
    "kb_context": state.get("kb_context", "")
    }