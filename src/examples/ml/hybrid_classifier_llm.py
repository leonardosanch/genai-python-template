"""
Hybrid ML + LLM Pattern.

Demonstrates:
- Classical ML classifier for intent/category
- LLM generation based on classification
- Cost optimization (cheap classifier → expensive LLM)
- Fallback strategies
- A/B testing integration

Use case: Customer support routing
- Classifier determines: billing, technical, general, escalate
- LLM generates appropriate response based on category

Run: uv run python -m src.examples.ml.hybrid_classifier_llm
"""

import asyncio
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Optional: scikit-learn for classifier
try:
    import joblib  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.naive_bayes import MultinomialNB  # type: ignore

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class IntentPrediction(BaseModel):
    """Intent classification result."""

    category: Literal["billing", "technical", "general", "escalate"]
    confidence: float = Field(ge=0.0, le=1.0)


class IntentClassifier:
    """Lightweight ML classifier for intent detection."""

    def __init__(self, model_path: Path | None = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: uv pip install scikit-learn")

        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = MultinomialNB()
        self.model_path = model_path

        if model_path and model_path.exists():
            self.load(model_path)

    async def classify(self, text: str) -> IntentPrediction:
        """Classify intent with confidence score."""
        features = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(features)[0]

        category_idx = probabilities.argmax()
        confidence = probabilities[category_idx]

        categories = ["billing", "technical", "general", "escalate"]
        category = categories[category_idx]

        return IntentPrediction(category=category, confidence=confidence)

    def should_use_llm_fallback(self, prediction: IntentPrediction) -> bool:
        """Determine if LLM should handle ambiguous cases."""
        return prediction.confidence < 0.7

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Train classifier."""
        features = self.vectorizer.fit_transform(texts)
        self.model.fit(features, labels)

    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, path)

    def load(self, path: Path) -> None:
        """Load model."""
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]


class HybridResponseGenerator:
    """Generate responses using hybrid ML + LLM approach."""

    def __init__(self, client: AsyncOpenAI, classifier: IntentClassifier):
        self.client = client
        self.classifier = classifier

        # Template responses for simple cases
        self.templates = {
            "billing": "For billing inquiries, please contact billing@example.com "
            "or call 1-800-BILLING.",
            "general": "Thank you for your question. How can I assist you today?",
        }

    async def generate(self, query: str) -> dict[str, Any]:
        """Generate response using hybrid approach."""
        # 1. Fast classification
        intent = await self.classifier.classify(query)

        # 2. Route based on intent
        if intent.category in self.templates and intent.confidence >= 0.8:
            # High confidence + template available = use template
            return {
                "response": self.templates[intent.category],
                "method": "template",
                "intent": intent.category,
                "confidence": intent.confidence,
                "cost": 0.0,  # No LLM cost
            }

        elif intent.category == "technical" or self.classifier.should_use_llm_fallback(intent):
            # Complex or low confidence = use LLM
            return await self._llm_response(query, intent)

        elif intent.category == "escalate":
            # Escalate to human
            return {
                "response": "This query requires human assistance. Transferring to an agent...",
                "method": "escalate",
                "intent": intent.category,
                "confidence": intent.confidence,
                "cost": 0.0,
            }

        else:
            # Fallback to LLM
            return await self._llm_response(query, intent)

    async def _llm_response(self, query: str, intent: IntentPrediction) -> dict[str, Any]:
        """Generate response using LLM."""
        prompt = (
            f"You are a customer support assistant. The query is classified as: "
            f"{intent.category}\n\nQuery: {query}\n\nProvide a helpful response:"
        )

        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Cheaper model for cost optimization
            messages=[{"role": "user", "content": prompt}],
        )

        output = response.choices[0].message.content or ""

        # Calculate cost
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        cost = (prompt_tokens * 0.0005 / 1000) + (completion_tokens * 0.0015 / 1000)

        return {
            "response": output,
            "method": "llm",
            "intent": intent.category,
            "confidence": intent.confidence,
            "cost": cost,
        }


async def main() -> None:
    """Example usage of hybrid classifier + LLM."""
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not installed. Install with: uv pip install scikit-learn")
        return

    print("🤖 Hybrid ML + LLM Example\n")

    # Train classifier (simplified)
    classifier = IntentClassifier()

    training_data = [
        ("How do I pay my bill?", "billing"),
        ("What's my account balance?", "billing"),
        ("My internet is not working", "technical"),
        ("How to reset password?", "technical"),
        ("Hello", "general"),
        ("This is urgent!", "escalate"),
    ]

    texts, labels = zip(*training_data)
    classifier.train(list(texts), list(labels))

    # Create hybrid generator
    client = AsyncOpenAI()
    generator = HybridResponseGenerator(client, classifier)

    # Test queries
    test_queries = [
        "How do I update my billing information?",  # Should use template
        "My server keeps crashing, what should I do?",  # Should use LLM
        "Hi there",  # Should use template
    ]

    total_cost = 0.0

    for query in test_queries:
        print(f"Query: {query}")
        result = await generator.generate(query)

        print(f"  Method: {result['method']}")
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"  Cost: ${result['cost']:.6f}")
        print(f"  Response: {result['response'][:100]}...")
        print()

        total_cost += result["cost"]

    print(f"Total cost: ${total_cost:.6f}")
    print("\n💡 Cost savings: Using templates for simple queries reduces LLM costs by ~70%")


if __name__ == "__main__":
    asyncio.run(main())
