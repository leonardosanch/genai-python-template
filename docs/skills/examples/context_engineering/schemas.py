from typing import Any

from pydantic import BaseModel, Field

# --- MCP Structured Messages ---


class MCPMessage(BaseModel):
    """Model Context Protocol message structure."""

    protocol_version: str = Field(default="1.0")
    sender: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Semantic Blueprint Schema ---


class Participant(BaseModel):
    """Entity participating in a semantic scene."""

    name: str
    role: str = Field(description="SRL Role: Agent, Patient, Recipient, Source, etc.")
    entity_type: str = Field(description="Person, Organization, System, etc.")


class SemanticBlueprint(BaseModel):
    """Level 5 Context: Semantic Blueprint."""

    scene_goal: str = Field(description="What this scene/step must achieve")
    participants: list[Participant]
    action_to_complete: str = Field(description="Predicate + arguments")
    modifiers: dict[str, str] = Field(description="Temporal, Locative, Manner modifiers")

    def to_prompt_context(self) -> str:
        """Render blueprint as context string for LLM."""
        participants_str = "\n".join(
            [f"- {p.name} ({p.role}): {p.entity_type}" for p in self.participants]
        )
        modifiers_str = "\n".join([f"- {k}: {v}" for k, v in self.modifiers.items()])

        return f"""
# Semantic Blueprint
**Goal**: {self.scene_goal}

**Participants**:
{participants_str}

**Action**: {self.action_to_complete}

**Context/Modifiers**:
{modifiers_str}
"""


# --- Input Sanitization Example ---


def helper_sanitize_input(user_input: str) -> str:
    """
    Basic input sanitization to prevent common prompt injection patterns.
    In production, use a more robust library or LLM-based guardrail.
    """
    # 1. Remove control characters
    sanitized = "".join(ch for ch in user_input if ch.isprintable())

    # 2. Block common injection keywords (naive list for example)
    denylist = ["ignore previous instructions", "system prompt", "delete all"]
    for term in denylist:
        if term.lower() in sanitized.lower():
            raise ValueError(f"Security: Blocked term '{term}' detected.")

    return sanitized


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Create a Blueprint
    blueprint = SemanticBlueprint(
        scene_goal="Determine eligibility for loan",
        participants=[
            Participant(name="User", role="Agent", entity_type="Customer"),
            Participant(name="BankSystem", role="Recipient", entity_type="System"),
        ],
        action_to_complete="Submit application data",
        modifiers={"Time": "Immediate", "Location": "Web Portal"},
    )

    print(blueprint.to_prompt_context())

    # 2. MCP Message
    msg = MCPMessage(
        sender="PlannerAgent", content="Requesting credit score check", metadata={"task_id": "123"}
    )
    print(msg.model_dump_json(indent=2))
