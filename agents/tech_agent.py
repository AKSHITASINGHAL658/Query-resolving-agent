from utils.llm import llm
from knowledge_base.tech_kb import TECH_KB

def technical_agent(state):
    """Handles application errors, installation issues, and login bugs."""
    query = state["messages"][-1].content
    state["kb_context"] = TECH_KB  # Save retrieved RAG content to state 
    # 🚨 AGENT EMERGENCY OVERWRITE
    if state.get("requires_approval") and not state.get("human_feedback"):
        state["query_category"] = "human_checkpoint"
        state["agent_draft_response"] = ""  # Strictly no summarization or draft generation
        return state
    #prompt
    prompt = f"""
    You are an expert Technical Support Engineer for ABC Technologies.
    Answer the customer's query using only the provided Technical Manual context.
    
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