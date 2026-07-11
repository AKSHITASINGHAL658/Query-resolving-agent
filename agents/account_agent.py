from utils.llm import llm
from knowledge_base.account_kb import ACCOUNT_KB

def account_agent(state):
    """Handles password resets, profile updates, and activation status."""
    query = state["messages"][-1].content
    state["kb_context"] = ACCOUNT_KB  # Save retrieved RAG content to state 
    # 🚨 AGENT EMERGENCY OVERWRITE
    if state.get("requires_approval") and not state.get("human_feedback"):
        state["query_category"] = "human_checkpoint"
        state["agent_draft_response"] = ""  # Strictly no summarization or draft generation
        return state
    #prompt
    prompt = f"""
    You are an expert Account Specialist for ABC Technologies
    Answer the customer's query using only the provided FAQ Document context.
    
    Context:
    {state["kb_context"]}
    
    Customer Query: {query}
    """
    
    result = llm.invoke(prompt)
    
    return {
        **state, 
        "agent_draft_response": result.content
    }