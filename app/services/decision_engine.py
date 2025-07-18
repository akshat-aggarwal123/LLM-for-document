import json
from dataclasses import dataclass
from typing import List, Optional
from .llama_model import Llama3Service

@dataclass
class DecisionResponse:
    decision: str
    amount: Optional[float] = None
    justification: str = ""
    relevant_clauses: List = None

class DecisionEngine:
    def __init__(self):
        self.llama = Llama3Service()

    def make_decision(self, query: str, entity, clauses) -> DecisionResponse:
        if not clauses:
            return DecisionResponse(decision="insufficient_information")

        context = "\n".join(
            f"Clause {i+1} (score={c.relevance_score:.2f}):\n{c.content}"
            for i, c in enumerate(clauses[:5])
        )
        raw = self.llama.decide(context, query)
        try:
            data = json.loads(raw)
            return DecisionResponse(
                decision=data.get("decision", "requires_review").lower(),
                amount=float(data["amount"]) if data.get("amount") else None,
                justification=data.get("justification", ""),
                relevant_clauses=clauses[:5],
            )
        except Exception:
            return DecisionResponse(
                decision="requires_review",
                justification="LLM malformed JSON",
                relevant_clauses=clauses[:5],
            )