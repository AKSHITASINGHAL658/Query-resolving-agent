from utils.llm import llm
from knowledge_base.account_kb import ACCOUNT_KB

def account_agent(state):
    """Handles password resets, profile updates, and activation status."""
    query = state["messages"][-1].content
    # Append (not overwrite) so multi-hop turns keep every consulted agent's context
    state["kb_context"] = (state.get("kb_context") or "") + f"\n\n[ACCOUNT KB]:\n{ACCOUNT_KB}"
    #prompt
    prompt = f"""
    You are an expert Account Specialist for ABC Technologies
    Answer the customer's query using only the provided FAQ Document context.
    
    Context:
    {ACCOUNT_KB}
    
    Customer Query: {query}
    """
    
    result = llm.invoke(prompt)
    
    return {
        **state, 
        "agent_draft_response": result.content
    }