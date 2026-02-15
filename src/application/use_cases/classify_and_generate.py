# src/application/use_cases/classify_and_generate.py
from dataclasses import dataclass

from src.domain.ports.llm_port import LLMPort
from src.domain.ports.ml_model_port import MLModelPort


@dataclass
class HybridResponse:
    intent: str
    response: str
    model_metadata: dict[str, str]


class ClassifyAndGenerateUseCase:
    """
    Hybrid pattern:
    1. Classify intent using a cheap/fast legacy ML model.
    2. Select specific system prompt based on classification.
    3. Generate response using LLM.
    """

    def __init__(self, classifier: MLModelPort[dict[str, float], str], llm: LLMPort):
        """
        Initialize use case.

        Args:
            classifier: The legacy classification model
            llm: The generative model
        """
        self.classifier = classifier
        self.llm = llm

    async def execute(self, features: dict[str, float], user_query: str) -> HybridResponse:
        """Execute the hybrid flow."""
        # Step 1: Deterministic / Cheap Classification
        intent = await self.classifier.predict(features)

        # Step 2: Prompt Routing
        system_prompt = self._get_system_prompt_for_intent(intent)

        # Step 3: Generative AI
        llm_response = await self.llm.generate(
            prompt=user_query, system_instruction=system_prompt, temperature=0.7
        )

        return HybridResponse(
            intent=intent, response=llm_response, model_metadata=self.classifier.metadata
        )

    def _get_system_prompt_for_intent(self, intent: str) -> str:
        """Select prompt strategy based on intent."""
        # Prompts split for line length compliance
        support = "You are a technical engineer. Provide step-by-step troubleshooting."
        sales = "You are a sales representative. Focus on benefits and closing the deal."

        prompts = {
            "refund": "You are a billing support agent. Be empathetic but strict about policy.",
            "technical_support": support,
            "sales": sales,
        }
        return prompts.get(intent, "You are a helpful assistant.")
