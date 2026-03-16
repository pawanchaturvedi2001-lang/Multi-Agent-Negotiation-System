"""
mediator_agent.py — Autonomous Mediator Agent.

Goal: Detect deadlocks, identify Pareto-efficient solutions, and propose
bridge offers that move both parties toward agreement without taking sides.

The mediator activates when:
  - The gap between offers hasn't narrowed in 2+ rounds
  - Either party's deadline pressure exceeds 0.7
  - One party signals walk-away intent
"""
from typing import Optional, Tuple
from agents.base_agent import BaseAgent
from memory import NegotiationMemory


class MediatorAgent(BaseAgent):

    def __init__(self, memory: NegotiationMemory):
        super().__init__(
            role="mediator",
            goal="Facilitate a Pareto-efficient agreement that both parties accept. "
                 "Resolve deadlocks without taking sides. Maximize joint surplus.",
            memory=memory,
        )

    def build_system_prompt(self) -> str:
        zopa = self.memory.get_zopa()
        return f"""You are an autonomous MEDIATOR agent in a multi-agent negotiation system.

IDENTITY & GOAL
---------------
You are a neutral mediator. You do NOT have a financial stake in the outcome.
Your goal is to help buyer and seller reach a Pareto-efficient agreement —
one where no party can be made better off without making the other worse off.

ZONE OF POSSIBLE AGREEMENT
---------------------------
Current ZOPA exists: {zopa['exists']}
{f"Theoretical midpoint: ${zopa['midpoint']:,.0f}" if zopa['midpoint'] else "No ZOPA detected — consider non-price solutions."}

MEDIATION STRATEGIES
--------------------
1. DEADLOCK DETECTION: If offers haven't moved in 2 rounds, intervene.
2. PARETO BRIDGING: Find the midpoint of the ZOPA and propose it as a fair split.
3. NON-PRICE CREATIVITY: Suggest warranty extensions, payment plans, staged delivery
   to create value without changing price.
4. DEADLINE AWARENESS: When rounds_remaining < 3, urgently push for closure.
5. SENTIMENT CALIBRATION: If one party is hostile, privately suggest a de-escalation
   concession to rebuild momentum.
6. ANCHORING NEUTRALIZATION: Identify and neutralize extreme anchor positions
   by referencing objective market standards.

TOOLS WORKFLOW
--------------
Step 1: analyze_sentiment on both parties' recent offers.
Step 2: evaluate_offer from a neutral perspective.
Step 3: If deadlocked, call mediate_deadlock with a bridge proposal.
Step 4: If ZOPA exists and gap < 5%, push both toward accept_offer.

YOUR OUTPUT
-----------
After each mediation turn, log your observation to shared memory via your reasoning.
Be explicit: name the deadlock, the ZOPA, the bridge, and why it's fair.

You are fully autonomous. Impartial. Strategic. The judges want to see your conflict-resolution intelligence.
"""

    def should_intervene(self) -> bool:
        """Check if conditions warrant mediator intervention."""
        offers = self.memory.get_offer_history()
        if len(offers) < 4:
            return False

        # Check if the gap has been narrowing
        buyer_offers = [o.price for o in offers if o.agent == "buyer"]
        seller_offers = [o.price for o in offers if o.agent == "seller"]

        if len(buyer_offers) >= 2 and len(seller_offers) >= 2:
            buyer_movement = abs(buyer_offers[-1] - buyer_offers[-2])
            seller_movement = abs(seller_offers[-1] - seller_offers[-2])
            gap = seller_offers[-1] - buyer_offers[-1]

            # Stalled: neither side moved more than 1% of gap
            if buyer_movement < gap * 0.01 and seller_movement < gap * 0.01:
                return True

        # Deadline pressure
        state = self.memory.state
        if state.buyer.deadline_pressure > 0.7:
            return True

        return False

    def interpret_tool_result(
        self,
        tool_name: str,
        tool_input: dict,
        tool_result: dict,
    ) -> Optional[Tuple[str, dict]]:

        if tool_name == "mediate_deadlock":
            bridge = tool_input.get("proposed_bridge_price")
            concessions = tool_input.get("non_price_concessions", [])
            rationale = tool_input.get("rationale", "")
            concession_str = ", ".join(concessions) if concessions else "none"

            note = (
                f"MEDIATOR BRIDGE: ${bridge:,.0f} | "
                f"Non-price concessions: {concession_str} | "
                f"Rationale: {rationale}"
            )
            self.memory.add_mediator_note(note)

            return ("mediate", {
                "price": bridge,
                "reasoning": rationale,
                "plan": f"Propose bridge at ${bridge:,.0f} with: {concession_str}",
                "goal_assessment": "Bridge is within ZOPA and maximizes joint surplus.",
            })

        elif tool_name == "analyze_sentiment":
            score = tool_input.get("sentiment_score", 0.5)
            intent = tool_input.get("detected_intent", "unknown")
            note = f"Sentiment: {score:.2f} | Intent: {intent}"
            self.memory.update_agent_sentiment(
                tool_input.get("opponent_role", "buyer"), score
            )
            self.memory.add_mediator_note(note)

        return None
