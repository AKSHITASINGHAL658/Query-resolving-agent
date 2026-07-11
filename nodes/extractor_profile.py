# Import LLM configuration
from utils.llm import llm

# Import JSON parser for structured extraction
import json


def extractor_profile(state):
    """
    Profile Extraction Node

    Responsibilities:
    1. Extract customer information from user messages.
    2. Identify customer name and customer ID.
    3. Store extracted details in workflow state.
    4. Provide customer identity data for memory,
       routing, and support agents.
    """

    # ============================================================
    # STEP 1: RETRIEVE LATEST USER MESSAGE
    # ============================================================

    # Get the most recent message from conversation history
    last_msg = state["messages"][-1].content.lower()

    # ============================================================
    # STEP 2: RULE-BASED NAME EXTRACTION
    # ============================================================

    # Quick extraction using simple pattern matching
    # Example:
    # "My name is Akshita"
    # -> customer_name = "Akshita"

    if "my name is" in last_msg:
        try:
            name = (
                last_msg
                .split("my name is")[1]
                .strip()
                .split()[0]
            )

            # Store formatted name in state
            state["customer_name"] = name.title()

        except:
            pass

    # ============================================================
    # STEP 3: LLM-BASED PROFILE EXTRACTION
    # ============================================================

    # Prompt instructing the LLM to return JSON only
    prompt = f"""
    Extract customer information.

    Message:
    {last_msg}

    Rules:
    - If information is missing return null
    - Never return 'unknown'
    - Return only JSON

    Format:

    {{
      "customer_name": null,
      "customer_id": null
    }}
    """

    # Send extraction task to LLM
    result = llm.invoke(prompt)

    # ============================================================
    # STEP 4: JSON PARSING & VALIDATION
    # ============================================================

    try:

        # Clean response text
        raw = result.content.strip()

        # Extract JSON block from model output
        start = raw.find("{")
        end = raw.rfind("}") + 1

        # Convert JSON string to Python dictionary
        data = json.loads(raw[start:end])

        # --------------------------------------------------------
        # Store Customer Name
        # --------------------------------------------------------
        if (
            data.get("customer_name")
            and data["customer_name"] != "unknown"
        ):
            state["customer_name"] = data["customer_name"]

        # --------------------------------------------------------
        # Store Customer ID
        # --------------------------------------------------------
        if (
            data.get("customer_id")
            and data["customer_id"] != "unknown"
        ):
            state["customer_id"] = data["customer_id"]

    except:
        # Ignore extraction failures and continue workflow
        pass

    # ============================================================
    # STEP 5: RETURN UPDATED STATE
    # ============================================================

    return state