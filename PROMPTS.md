# Prompt Engineering

## Principios

1. **Prompts son artefactos versionados**, no strings hardcodeados
2. **Separar prompt de lógica de negocio**
3. **Structured output siempre** (Pydantic schemas)
4. **Determinismo cuando sea posible** (temperature=0, seed)
5. **Evaluar prompts sistemáticamente**, no a ojo

---

## Versionado de Prompts

Los prompts se almacenan como templates con versión explícita.

```python
# src/domain/prompts/summarize.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SummarizePrompt:
    version: str = "1.2.0"
    template: str = """You are a document summarization expert.

Given the following context, provide a structured summary.

## Context
{context}

## Query
{query}

## Instructions
- Extract key points (max {max_points})
- Include confidence score (0-1)
- Cite sources when available
- Respond in JSON format matching the provided schema
"""

    def render(self, query: str, context: str, max_points: int = 5) -> str:
        return self.template.format(
            query=query, context=context, max_points=max_points
        )
```

---

## Templates

### System Prompts

```python
SYSTEM_PROMPTS = {
    "researcher": """You are a research assistant specialized in {domain}.
Your goal is to find accurate, up-to-date information.
Always cite your sources. If uncertain, say so explicitly.""",

    "code_reviewer": """You are a senior code reviewer.
Focus on: security vulnerabilities, performance issues, maintainability.
Be concise. Use bullet points. Suggest specific fixes.""",
}
```

### Prompt Composition

```python
def build_rag_prompt(
    query: str,
    context_docs: list[Document],
    system_prompt: str | None = None,
) -> list[dict]:
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.content}"
        for doc in context_docs
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })
    return messages
```

---

## Structured Output con Pydantic

```python
from pydantic import BaseModel, Field

class Summary(BaseModel):
    title: str = Field(description="Brief title of the summary")
    key_points: list[str] = Field(description="Main points extracted")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    sources: list[str] = Field(default_factory=list, description="Cited sources")

# Con Instructor
import instructor
from openai import AsyncOpenAI

client = instructor.from_openai(AsyncOpenAI())

summary = await client.chat.completions.create(
    model="gpt-4o",
    response_model=Summary,
    messages=messages,
)
# summary es un objeto Summary validado por Pydantic
```

---

## Técnicas de Prompt Engineering

### Chain of Thought (CoT)

```python
COT_SUFFIX = """
Think step by step:
1. First, identify the key entities
2. Then, analyze their relationships
3. Finally, provide your answer with reasoning
"""
```

### Few-Shot

```python
FEW_SHOT_EXAMPLES = """
Example 1:
Input: "Python web frameworks"
Output: {"category": "technology", "subcategory": "web", "language": "python"}

Example 2:
Input: "Machine learning with TensorFlow"
Output: {"category": "technology", "subcategory": "ml", "language": "python"}
"""
```

### Self-Consistency

Generar múltiples respuestas y seleccionar por consenso:

```python
async def self_consistent_generate(llm: LLMPort, prompt: str, n: int = 3) -> str:
    responses = await asyncio.gather(
        *[llm.generate(prompt, temperature=0.7) for _ in range(n)]
    )
    return select_majority(responses)
```

---

## Evaluación de Prompts

```python
SUMMARY_TEST_CASES = [
    {
        "input": "What are the benefits of RAG?",
        "context": "RAG improves accuracy by grounding responses...",
        "expected_fields": ["title", "key_points", "confidence"],
        "min_confidence": 0.7,
    },
]

@pytest.mark.parametrize("case", SUMMARY_TEST_CASES)
async def test_summary_prompt_quality(case):
    result = await generate_summary(case["input"], case["context"])
    for field in case["expected_fields"]:
        assert getattr(result, field) is not None
    assert result.confidence >= case["min_confidence"]
```

---

## Reglas

1. **No hardcodear prompts en application/infrastructure** — mantenerlos en domain/prompts/
2. **Versionar cada cambio** — un cambio de prompt es un cambio de comportamiento
3. **Testear cada prompt** — ver [TESTING.md](TESTING.md)
4. **No exponer system prompts** al usuario — ver [SECURITY.md](SECURITY.md)
5. **Usar structured output** siempre que el output tenga schema predecible

Ver también: [EVALUATION.md](EVALUATION.md), [TESTING.md](TESTING.md)
