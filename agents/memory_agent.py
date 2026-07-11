from langchain_core.messages import AIMessage

def memory_agent(state: dict) -> dict:
    """
    Task 7 & 10 (Query 5): Bypasses standard routing to recall past session history.
    Uses the historical messages automatically fetched by the database checkpointer.
    """
    # 1. Grab the extracted customer name and ID from the State structure
    customer_name = state.get("customer_name")
    customer_id = state.get("customer_id")
    
    # LangGraph's checkpointer automatically loads the conversation 
    # history into state["messages"] for the active thread_id.
    messages = state.get("messages", [])
    previous_issue = None

    # Scan backward through the history (skipping the current query at the end)
    # to locate the customer's actual historical issue.
    for msg in reversed(messages[:-1]):
        if msg.type == "human" or getattr(msg, "role", "") == "user":
            content_lower = msg.content.lower()
            # Ignore meta-queries about history to find the actual problem statement
            if not any(word in content_lower for word in ["previous", "history", "last time", "what was my"]):
                previous_issue = msg.content
                break

    # 2. Build the greeting prefix dynamically depending on whether a name was saved
    greeting = f"Hi {customer_name} with customer ID {customer_id} we want to tell you that, y" if customer_name else "Y"

    # 3. Construct the response string matching your evaluation requirements
    if previous_issue:
        answer = f"{greeting}our previous support issue was: \"{previous_issue}\" we are stating this as you requested"
    else:
        answer = f"{greeting}our conversation history doesn't contain any previous support requests recorded under this session."

    # 4. Format the final output precisely as a tagged string context for the supervisor
    formatted_response = f"[Previous Issue Raised]: {answer}"

    # 5. Return the clean string payload directly back into the state channel
    return {
        "agent_draft_response": formatted_response,
        "query_category": "supervisor"  # Route to supervisor cleanly without type errors
    }