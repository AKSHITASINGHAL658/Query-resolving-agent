# Typing & schema tools for structured router output
from typing import Literal
from pydantic import BaseModel, Field

# Import AIMessage to append final responses into conversation history
from langchain_core.messages import AIMessage

# Import configured LLM instance
from utils.llm import llm

# Rich components for terminal visualization
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.console import Group

# Initialize Rich console
console = Console()


# ================================================================
# PART 1: SUPERVISOR ROUTER (LLM ROUTING BRAIN)
# ================================================================

# Specialized agents the router is allowed to hand off to
AGENT_NAMES = ["sales", "technical", "billing", "account", "memory"]

# Loop guard: maximum passes through supervisor_router per turn.
# Once exceeded, FINISH is forced so a confused LLM cannot loop forever.
MAX_ROUTING_HOPS = 3

AGENT_DESCRIPTIONS = """
- sales: pre-purchase questions only -- product information, subscription plans, pricing, features, buying or upgrading.
  Sales does NOT handle refunds, money back, or disputes about past charges, even if a plan or price is mentioned.
- technical: app crashes, errors, bugs, installation, login failures, uploads, configuration.
- billing: owns ALL refund requests and refund policy, plus invoices, payments, charges, receipts, transactions.
- account: password resets, profile updates, activation/deactivation, usernames, account closure.
- memory: ONLY when the customer asks about their previous support issue or conversation history.
"""


class RouterDecision(BaseModel):
    """Structured routing decision returned by the supervisor router LLM."""

    # Reasoning comes first so the model thinks before it commits to `next`
    reasoning: str = Field(
        description="One or two sentences explaining which parts of the "
                    "customer's request are covered and what is still missing."
    )
    next: Literal["sales", "technical", "billing", "account", "memory", "FINISH"] = Field(
        description="The specialist agent to call next, or FINISH when every "
                    "part of the customer's request has been addressed."
    )


# LLM constrained to emit a RouterDecision
router_llm = llm.with_structured_output(RouterDecision)


def supervisor_router(state: dict) -> dict:
    """
    Supervisor Router (LLM-based)

    Responsibilities:
    1. Collect the report-back from the agent that just ran.
    2. Ask the LLM which specialist to call next (or FINISH),
       given the query, conversation so far, agent responses
       and the requires_approval HITL flag.
    3. Enforce the loop guard (hop cap + no repeat agents).
    4. On FINISH, combine all agent responses into one draft
       for the supervisor_agent reviewer to polish.
    """

    # ============================================================
    # STEP 1: COLLECT REPORT-BACK FROM PREVIOUS AGENT
    # ============================================================

    hops = state.get("routing_hops", 0)
    agent_responses = list(state.get("agent_responses") or [])

    # On every pass after the first, the agent we dispatched last
    # time has just written its draft -- record it
    last_agent = state.get("next_agent")
    if hops > 0 and last_agent in AGENT_NAMES:
        agent_responses.append({
            "agent": last_agent,
            "response": state.get("agent_draft_response") or ""
        })

    hops += 1
    consulted = [r["agent"] for r in agent_responses]

    # ============================================================
    # STEP 2: LOOP GUARD OR LLM ROUTING DECISION
    # ============================================================

    if hops > MAX_ROUTING_HOPS:

        next_agent = "FINISH"
        reasoning = (
            f"Loop guard: routing hop cap ({MAX_ROUTING_HOPS}) "
            f"reached, forcing FINISH."
        )

    else:

        query = state["messages"][-1].content

        # Earlier turns of the conversation (excluding current query)
        history = "\n".join(
            f"{m.type.upper()}: {m.content}"
            for m in state.get("messages", [])[-7:-1]
        ) or "(no earlier conversation)"

        responses_text = "\n\n".join(
            f"[{r['agent'].upper()} AGENT]:\n{r['response'][:800]}"
            for r in agent_responses
        ) or "(no agents consulted yet)"

        approval_note = (
            "YES - this request is sensitive (refund/cancellation/"
            "escalation). A human will approve the final reply, so "
            "make sure the specialist that owns the sensitive action "
            "has been consulted before finishing."
            if state.get("requires_approval")
            else "NO"
        )

        router_prompt = f"""
        You are the Supervisor Router for ABC Technologies customer support.
        You do not answer the customer yourself. You decide which specialist
        agent should handle the request next, one at a time, and you decide
        when the request is fully handled.

        Available agents:
        {AGENT_DESCRIPTIONS}

        Rules:
        - A single request can need more than one agent. Look for every
          distinct part of the customer's message (e.g. a billing request
          AND a technical problem) and make sure each part is handled by
          the right specialist.
        - Read the agent responses so far. If part of the request is still
          unaddressed, route to the specialist who owns that part.
        - Never choose an agent that has already been consulted.
        - Only choose an agent whose area is actually mentioned in the
          customer's message.
        - Route by the specific ACTION the customer wants, not by adjacent
          keywords. Any refund request goes to billing, whatever else the
          message mentions.
        - Human approval for sensitive requests happens automatically
          AFTER you choose FINISH. Do not route to another agent to get
          approval.
        - Choose FINISH as soon as every part of the request is addressed.

        Conversation so far:
        {history}

        Current customer query: {query}

        Requires human approval: {approval_note}

        Agents already consulted: {", ".join(consulted) or "none"}

        Agent responses so far:
        {responses_text}

        Routing pass {hops} of {MAX_ROUTING_HOPS}.
        """

        with console.status(
            "[bold magenta]Supervisor Router is deciding next agent...",
            spinner="dots"
        ):
            try:
                decision = router_llm.invoke(router_prompt)
                next_agent = decision.next
                reasoning = decision.reasoning
            except Exception as e:
                # Malformed structured output: fall back safely
                next_agent = "FINISH" if agent_responses else "account"
                reasoning = f"Router LLM failed ({type(e).__name__}); fallback route."

        # Guard: re-calling the same agent on the same query adds nothing
        if next_agent in consulted:
            reasoning += f" [Guard: '{next_agent}' already consulted -> FINISH]"
            next_agent = "FINISH"

        # Guard: never finish before at least one specialist has answered
        if next_agent == "FINISH" and not agent_responses:
            reasoning += " [Guard: no specialist consulted yet -> account]"
            next_agent = "account"

    # ============================================================
    # STEP 3: VISUAL AUDIT PANEL
    # ============================================================

    router_text = Text()
    router_text.append("🔁 Routing Pass: ", style="bold white")
    router_text.append(f"{hops} / {MAX_ROUTING_HOPS}\n", style="bold cyan")
    router_text.append("📋 Agents Consulted: ", style="bold white")
    router_text.append(f"{' → '.join(consulted) or 'none'}\n", style="yellow")
    router_text.append("🧠 Reasoning: ", style="bold white")
    router_text.append(f"{reasoning}\n", style="italic")
    router_text.append("🔀 Next: ", style="bold white")
    router_text.append(
        next_agent.upper(),
        style="bold green underline" if next_agent == "FINISH" else "bold magenta underline"
    )

    console.print(
        Panel(
            router_text,
            title="🧭 LLM Supervisor Router",
            border_style="magenta",
            expand=False
        )
    )

    # ============================================================
    # STEP 4: STATE UPDATE
    # ============================================================

    update = {
        "next_agent": next_agent,
        "router_reasoning": reasoning,
        "routing_hops": hops,
        "agent_responses": agent_responses,
    }

    # On FINISH, merge every agent's report into a single draft
    # for the supervisor_agent reviewer
    if next_agent == "FINISH":

        if len(agent_responses) == 1:
            combined = agent_responses[0]["response"]
        else:
            combined = "\n\n".join(
                f"[{r['agent'].title()} Team]: {r['response']}"
                for r in agent_responses
            )

        update["agent_draft_response"] = combined
        update["query_category"] = " + ".join(consulted)

    return update


# ================================================================
# PART 2: SUPERVISOR AGENT (RESPONSE REVIEWER + HITL)
# ================================================================


def supervisor_agent(state: dict) -> dict:
    """
    Supervisor Agent

    Responsibilities:
    1. Retrieve customer identity and memory information.
    2. Review responses generated by specialized agents.
    3. Handle Human-in-the-Loop (HITL) approvals/rejections.
    4. Refine responses using the LLM.
    5. Produce customer-ready professional responses.
    6. Update workflow state and route execution.
    """

    # ============================================================
    # STEP 1: MEMORY & CUSTOMER PROFILE RECALL
    # ============================================================

    # Retrieve customer details stored in memory
    customer_name = state.get("customer_name", "Anonymous User")
    customer_id = state.get("customer_id", "Not Assigned")

    # Create visual proof showing memory retrieval
    memory_proof_text = Text()
    memory_proof_text.append("👤 Identity Verified: ", style="bold white")
    memory_proof_text.append(f"{customer_name}\n", style="bold green")

    memory_proof_text.append("🆔 Profile ID: ", style="bold white")
    memory_proof_text.append(f"{customer_id}\n", style="bold cyan")

    memory_proof_text.append("📚 Memory Status: ", style="bold white")
    memory_proof_text.append(
        "ACTIVE - PERSISTED TO SQLITE",
        style="italic yellow"
    )

    # Display memory retrieval panel
    console.print("\n")
    console.print(
        Panel(
            memory_proof_text,
            title="💾 SQLITE MEMORY RECALL RECORD PROOF",
            border_style="magenta",
            expand=False
        )
    )

    # ============================================================
    # STEP 2: LOAD CURRENT WORKFLOW STATE
    # ============================================================

    # Draft generated by specialized agent
    draft = state.get("agent_draft_response", "")

    # Latest customer query
    query = state["messages"][-1].content

    # Indicates whether approval is required
    is_sensitive = state.get("requires_approval", False)

    # Human reviewer feedback (approval/rejection)
    human_feedback = state.get("human_feedback", None)

    # ============================================================
    # STEP 3: HUMAN-IN-THE-LOOP PROCESSING
    # ============================================================

    if human_feedback:

        # Create audit trail text
        text_to_polish = (
            f"[Human Decision]: {human_feedback.upper()}"
        )

        # --------------------------------------------------------
        # CASE A: REQUEST APPROVED
        # --------------------------------------------------------
        if human_feedback == "approved":

            # Build prompt for approval scenario
            supervisor_prompt = f"""
            You are a Senior Customer Support Supervisor at
            ABC Technologies.

            Management has APPROVED the customer's request.

            Your job is to create a professional final response
            informing the customer of the approved resolution.

            Customer Name: {customer_name}
            Customer ID: {customer_id}
            Original Request: {query}
            Approved Resolution: {draft}
            """

        # --------------------------------------------------------
        # CASE B: REQUEST REJECTED
        # --------------------------------------------------------
        else:

            # Build prompt for rejection scenario
            supervisor_prompt = f"""
            You are a Senior Customer Support Supervisor at
            ABC Technologies.

            Management has REJECTED the customer's request.

            Draft a professional and empathetic response
            explaining the reason for rejection.

            Customer Name: {customer_name}
            Customer ID: {customer_id}
            Original Request: {query}
            Reason for Rejection: {human_feedback}
            """

        # Generate final customer-facing response
        with console.status(
            "[bold green]Supervisor is crafting final response...",
            spinner="dots"
        ):
            polished_response = (
                llm.invoke(supervisor_prompt)
                .content
                .strip()
            )

    # ============================================================
    # STEP 4: NORMAL EXECUTION PATH
    # ============================================================
    else:

        # Retrieve RAG context if available
        kb_context = state.get("kb_context", "")

        # Current query classification
        query_category = state.get("query_category", "")

        # Display retrieved knowledge base evidence
        if kb_context:
            console.print("\n")
            console.print(
                Panel(
                    f"[bold white]{kb_context.strip()}[/bold white]",
                    title=(
                        f"🔍 RAG KNOWLEDGE BASE "
                        f"RETRIEVAL EVIDENCE "
                        f"({query_category.upper()})"
                    ),
                    border_style="cyan",
                    expand=False
                )
            )

        # Use draft if available, otherwise use query
        text_to_polish = (
            draft if draft.strip() else query
        )

        # Prompt instructing supervisor to polish response
        supervisor_prompt = f"""
        You are a Senior Customer Support Supervisor.

        Rewrite the following response into a polished,
        professional customer-ready format. It may contain
        input from several specialist teams; merge it into
        one coherent reply that addresses every point.

        Customer Name: {customer_name}
        Customer ID: {customer_id}

        Response:
        {text_to_polish}
        """

        # Generate refined response
        with console.status(
            "[bold green]Supervisor is polishing response...",
            spinner="dots"
        ):
            polished_response = (
                llm.invoke(supervisor_prompt)
                .content
                .strip()
            )

    # ============================================================
    # STEP 5: VISUAL AUDIT PANELS
    # ============================================================

    # Build panel showing raw agent response
    agent_text = Text()

    if human_feedback:

        status_color = (
            "green"
            if human_feedback == "approved"
            else "red"
        )

        agent_text.append(
            "🕵️ Human Operator Status: ",
            style="bold white"
        )

        agent_text.append(
            f"{human_feedback.upper()}\n\n",
            style=f"bold {status_color}"
        )

    agent_text.append(
        "❓ Current Working Message: ",
        style="bold cyan"
    )

    agent_text.append(
        f"{text_to_polish}\n\n",
        style="italic"
    )

    agent_panel = Panel(
        agent_text,
        title="🤖 Raw Unsupervised Agent Response",
        border_style="yellow",
        expand=False
    )

    # Determine next workflow route
    router_target = (
        "🔴 HUMAN CHECKPOINT"
        if is_sensitive and not human_feedback
        else "🟢 END"
    )

    # Build supervisor output panel
    supervisor_text = Text()

    supervisor_text.append(
        "✨ Polished Response Sent to Customer:\n",
        style="bold green"
    )

    supervisor_text.append(
        polished_response,
        style="bold white"
    )

    supervisor_text.append(
        "\n\n🔗 Supervisor Next Routing Decision: ",
        style="bold magenta"
    )

    supervisor_text.append(
        router_target,
        style="underline"
    )

    supervisor_panel = Panel(
        supervisor_text,
        title="🛡️ Supervisor Final Response",
        border_style="green",
        expand=False
    )

    # Display both panels together
    console.print(
        Group(
            agent_panel,
            supervisor_panel
        )
    )

    console.print("\n")

    # ============================================================
    # STEP 6: STATE MANAGEMENT & ROUTING
    # ============================================================

    # Sensitive requests go to HITL checkpoint
    if is_sensitive and not human_feedback:

        return {
            "agent_draft_response": polished_response,
            "query_category": "human_checkpoint"
        }

    # Normal completion path
    else:

        current_messages = state.get(
            "messages",
            []
        )

        return {
            "messages": current_messages + [
                AIMessage(content=polished_response)
            ],
            "agent_draft_response": "",
            "human_feedback": None,
            "requires_approval": False,
            "query_category": "end"
        }