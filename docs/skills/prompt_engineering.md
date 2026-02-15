# Skill: Prompt Engineering & Versioning

## Description

Production-grade prompt engineering practices including versioning, templating, testing, and systematic evaluation. This skill covers treating prompts as code artifacts with proper lifecycle management.

## Executive Summary

**Critical rules (always enforce):**
- **Prompts are versioned artifacts** — NEVER hardcode prompts in application code
- **Use templates with typed variables** — Jinja2 + Pydantic for validation and reusability
- **Separate concerns** — System prompt, user prompt, and few-shot examples are distinct artifacts
- **Evaluate before deploy** — Test prompts with metrics (accuracy, latency, cost) before production
- **Maintain prompt registry** — Track version, author, performance metrics, and deployment status
- **Version control prompts** — Store in Git, use semantic versioning (v1.0.0, v1.1.0, etc.)
- **A/B test prompt changes** — Never replace production prompts without comparative evaluation

**Read full skill when:** Building production LLM applications, implementing prompt optimization workflows, debugging quality issues, or establishing prompt governance.

---

## Deep Dive

## Core Concepts

1. **Prompt as Code**
   - Prompts are first-class artifacts, not magic strings
   - Stored in files, versioned in Git, reviewed in PRs
   - Deployed through CI/CD with testing gates
   - Rollback capability when performance degrades

2. **Template Variables**
   - Prompts have typed inputs (Pydantic models)
   - Variables injected at runtime via Jinja2
   - Validation prevents malformed prompts
   - Reusability across similar use cases

3. **Few-Shot Learning**
   - Examples stored separately from prompt template
   - Dynamic example selection based on input
   - Example quality > quantity (3-5 good examples often sufficient)
   - Examples versioned alongside prompts

4. **Chain-of-Thought (CoT)**
   - Explicit reasoning steps improve accuracy
   - "Let's think step by step" pattern
   - Structured CoT with XML/JSON formatting
   - Self-consistency via multiple CoT samples

5. **Prompt Testing**
   - Unit tests for prompt structure (variables, formatting)
   - Integration tests for LLM outputs (metrics-based)
   - Regression tests against golden datasets
   - A/B tests for production deployment

---

## External Resources

### 📚 Guides & Documentation

- **[OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)**  
  *Best for:* Official best practices and patterns
  
- **[Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library)**  
  *Best for:* Production-ready prompt templates
  
- **[LangChain Hub](https://smith.langchain.com/hub)**  
  *Best for:* Community-contributed prompts with versioning
  
- **[PromptLayer](https://promptlayer.com/)**  
  *Best for:* Prompt management and analytics

### 🔬 Research

- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)** (Wei et al., 2022)  
  *Best for:* Understanding reasoning improvements with CoT
  
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)** (Wang et al., 2022)  
  *Best for:* Improving accuracy via sampling and voting
  
- **[Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)** (Yao et al., 2023)  
  *Best for:* Complex problem-solving with search
  
- **[ReAct Pattern](https://arxiv.org/abs/2210.03629)** (Yao et al., 2022)  
  *Best for:* Combining reasoning and acting (tool use)

### 🛠️ Tools

- **[LangSmith](https://docs.smith.langchain.com/)**  
  *Best for:* Prompt versioning, testing, and monitoring
  
- **[PromptLayer](https://promptlayer.com/)**  
  *Best for:* Prompt analytics and version control
  
- **[Humanloop](https://humanloop.com/)**  
  *Best for:* Collaborative prompt engineering with evaluation
  
- **[Weights & Biases Prompts](https://wandb.ai/site/prompts)**  
  *Best for:* Experiment tracking for prompt optimization

---

## Prompt Template Architecture

### Directory Structure

```
prompts/
├── v1/
│   ├── summarization/
│   │   ├── system.txt              # System prompt
│   │   ├── user.jinja2             # User prompt template
│   │   ├── examples.json           # Few-shot examples
│   │   └── metadata.json           # Version metadata
│   ├── extraction/
│   │   ├── system.txt
│   │   ├── user.jinja2
│   │   ├── examples.json
│   │   └── metadata.json
│   └── classification/
│       └── ...
├── v2/
│   └── summarization/              # Improved version
│       ├── system.txt
│       ├── user.jinja2
│       ├── examples.json
│       └── metadata.json
├── registry.json                    # Central registry
└── evaluations/
    ├── summarization_v1_results.json
    └── summarization_v2_results.json
```

### Template Example (Jinja2 + Pydantic)

```python
from pathlib import Path
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


class SummarizationInput(BaseModel):
    """Input schema for summarization prompt."""
    
    text: str = Field(description="Text to summarize")
    max_words: int = Field(default=100, ge=10, le=500)
    style: str = Field(
        default="professional",
        pattern="^(professional|casual|technical)$",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific topics to emphasize",
    )


class PromptTemplate:
    """Prompt template with Jinja2 + Pydantic validation."""
    
    def __init__(
        self,
        template_dir: Path,
        version: str = "v1",
        task: str = "summarization",
    ):
        self.template_dir = template_dir
        self.version = version
        self.task = task
        
        # Load Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(template_dir / version / task),
            autoescape=False,
        )
        
        # Load system prompt
        system_path = template_dir / version / task / "system.txt"
        self.system_prompt = system_path.read_text() if system_path.exists() else ""
        
        # Load user template
        self.user_template = self.env.get_template("user.jinja2")
        
        # Load examples
        examples_path = template_dir / version / task / "examples.json"
        if examples_path.exists():
            import json
            self.examples = json.loads(examples_path.read_text())
        else:
            self.examples = []
    
    def render(self, input_data: SummarizationInput) -> dict[str, str]:
        """
        Render prompt with validated input.
        
        Args:
            input_data: Validated input following schema
            
        Returns:
            Dict with system and user prompts
        """
        # Render user prompt with variables
        user_prompt = self.user_template.render(
            text=input_data.text,
            max_words=input_data.max_words,
            style=input_data.style,
            focus_areas=input_data.focus_areas,
            examples=self.examples[:3],  # Include top 3 examples
        )
        
        return {
            "system": self.system_prompt,
            "user": user_prompt,
        }


# Example usage
def main() -> None:
    template = PromptTemplate(
        template_dir=Path("prompts"),
        version="v1",
        task="summarization",
    )
    
    input_data = SummarizationInput(
        text="Long article text here...",
        max_words=150,
        style="professional",
        focus_areas=["key findings", "methodology"],
    )
    
    prompts = template.render(input_data)
    print("System:", prompts["system"])
    print("User:", prompts["user"])
```

**Example `user.jinja2` template:**

```jinja2
Summarize the following text in {{ max_words }} words or less.

Style: {{ style }}
{% if focus_areas %}
Focus on: {{ focus_areas | join(", ") }}
{% endif %}

{% if examples %}
Examples:
{% for example in examples %}
Input: {{ example.input }}
Output: {{ example.output }}

{% endfor %}
{% endif %}

Text to summarize:
{{ text }}

Summary:
```

### Registry Schema

```python
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Literal


class PromptMetrics(BaseModel):
    """Performance metrics for a prompt version."""
    
    accuracy: float | None = Field(None, ge=0.0, le=1.0)
    latency_p50_ms: float | None = Field(None, ge=0.0)
    latency_p95_ms: float | None = Field(None, ge=0.0)
    cost_per_request: float | None = Field(None, ge=0.0)
    hallucination_rate: float | None = Field(None, ge=0.0, le=1.0)
    user_satisfaction: float | None = Field(None, ge=0.0, le=5.0)


class PromptVersion(BaseModel):
    """Metadata for a specific prompt version."""
    
    version: str = Field(pattern=r"^v\d+$", description="Version identifier (e.g., v1, v2)")
    task: str = Field(description="Task name (e.g., summarization, extraction)")
    template_path: Path = Field(description="Path to template directory")
    variables: list[str] = Field(description="Required template variables")
    model: str = Field(description="Target LLM model (e.g., gpt-4-turbo)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)
    
    created_at: datetime = Field(default_factory=datetime.now)
    author: str = Field(description="Prompt author")
    description: str = Field(description="What changed in this version")
    
    metrics: PromptMetrics | None = None
    status: Literal["draft", "testing", "production", "deprecated"] = "draft"
    
    # A/B testing
    traffic_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Links
    evaluation_results_url: HttpUrl | None = None
    git_commit: str | None = Field(None, pattern=r"^[a-f0-9]{40}$")


class PromptRegistry(BaseModel):
    """Central registry of all prompt versions."""
    
    prompts: dict[str, list[PromptVersion]] = Field(
        default_factory=dict,
        description="Mapping of task name to versions",
    )
    
    def get_production_version(self, task: str) -> PromptVersion | None:
        """Get the current production version for a task."""
        versions = self.prompts.get(task, [])
        production = [v for v in versions if v.status == "production"]
        return production[0] if production else None
    
    def get_version(self, task: str, version: str) -> PromptVersion | None:
        """Get a specific version."""
        versions = self.prompts.get(task, [])
        return next((v for v in versions if v.version == version), None)
    
    def add_version(self, version: PromptVersion) -> None:
        """Add a new prompt version."""
        if version.task not in self.prompts:
            self.prompts[version.task] = []
        self.prompts[version.task].append(version)
    
    def promote_to_production(self, task: str, version: str) -> None:
        """Promote a version to production (demote others)."""
        versions = self.prompts.get(task, [])
        for v in versions:
            if v.version == version:
                v.status = "production"
                v.traffic_percentage = 100.0
            elif v.status == "production":
                v.status = "deprecated"
                v.traffic_percentage = 0.0
    
    def save(self, path: Path) -> None:
        """Save registry to JSON file."""
        path.write_text(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "PromptRegistry":
        """Load registry from JSON file."""
        import json
        data = json.loads(path.read_text())
        return cls(**data)
```

---

## Prompt Patterns

### 1. Zero-Shot

**Description:** No examples provided, rely on model's pre-training.

**When to use:** Simple, well-defined tasks; when examples are unavailable; for general knowledge queries.

**Example:**

```python
system_prompt = """
You are a helpful assistant that answers questions concisely and accurately.
"""

user_prompt = """
What is the capital of France?
"""
```

**Pros:** Simple, fast, no example curation needed  
**Cons:** Lower accuracy on complex or domain-specific tasks

---

### 2. Few-Shot

**Description:** Provide 3-5 examples to guide the model's output format and style.

**When to use:** Structured outputs, domain-specific tasks, consistent formatting required.

**Example:**

```python
system_prompt = """
You are a sentiment classifier. Classify text as positive, negative, or neutral.
"""

user_prompt = """
Examples:
Input: "I love this product!"
Output: positive

Input: "This is terrible."
Output: negative

Input: "It's okay, nothing special."
Output: neutral

Now classify:
Input: "Best purchase I've ever made!"
Output:
"""
```

**Pros:** Significantly improves accuracy, guides output format  
**Cons:** Requires curated examples, increases token usage

---

### 3. Chain-of-Thought (CoT)

**Description:** Prompt model to show reasoning steps before final answer.

**When to use:** Math problems, logical reasoning, multi-step tasks, debugging.

**Example:**

```python
user_prompt = """
Question: A store has 15 apples. They sell 7 and then receive a shipment of 12 more. How many apples do they have now?

Let's solve this step by step:
1. Starting apples: 15
2. After selling 7: 15 - 7 = 8
3. After receiving 12: 8 + 12 = 20

Answer: 20 apples

Question: A train travels 60 km in 45 minutes. What is its speed in km/h?

Let's solve this step by step:
"""
```

**Pros:** Dramatically improves reasoning accuracy, provides explainability  
**Cons:** Increases latency and cost, requires more tokens

---

### 4. ReAct (Reasoning + Acting)

**Description:** Interleave reasoning, action (tool use), and observation in a loop.

**When to use:** Agents with tool access, multi-step workflows, external data needed.

**Example:**

```python
system_prompt = """
You solve problems by reasoning and using tools. Follow this format:

Thought: [your reasoning]
Action: [tool name and input]
Observation: [tool output]
... (repeat as needed)
Thought: I now know the final answer
Final Answer: [your answer]

Available tools:
- search(query): Search the web
- calculate(expression): Evaluate math expression
"""

user_prompt = """
Question: What is the population of the capital of France?

Thought: I need to find the capital of France first.
Action: search("capital of France")
Observation: The capital of France is Paris.
Thought: Now I need the population of Paris.
Action: search("population of Paris 2024")
Observation: Paris has approximately 2.2 million people.
Thought: I now know the final answer.
Final Answer: The population of Paris is approximately 2.2 million people.
"""
```

**Pros:** Enables complex multi-step reasoning with tools  
**Cons:** Complex to implement, requires tool integration, higher latency

---

### 5. Self-Ask

**Description:** Model decomposes question into sub-questions and answers them sequentially.

**When to use:** Complex questions requiring multiple facts, research-style queries.

**Example:**

```python
user_prompt = """
Question: Who was president of the US when the first iPhone was released?

Are follow-up questions needed here: Yes.
Follow-up: When was the first iPhone released?
Intermediate answer: The first iPhone was released on June 29, 2007.
Follow-up: Who was president of the US in June 2007?
Intermediate answer: George W. Bush was president in June 2007.
So the final answer is: George W. Bush

Question: What is the population of the birthplace of the inventor of the telephone?

Are follow-up questions needed here:
"""
```

**Pros:** Breaks down complex questions, improves factual accuracy  
**Cons:** Increases latency, may generate unnecessary sub-questions

---

## Prompt Testing Framework

### Unit Tests for Prompts

```python
import pytest
from pathlib import Path


def test_prompt_template_renders_correctly():
    """Test that prompt template renders with valid input."""
    template = PromptTemplate(
        template_dir=Path("prompts"),
        version="v1",
        task="summarization",
    )
    
    input_data = SummarizationInput(
        text="Test text",
        max_words=50,
        style="professional",
    )
    
    prompts = template.render(input_data)
    
    assert "system" in prompts
    assert "user" in prompts
    assert "Test text" in prompts["user"]
    assert "50 words" in prompts["user"]


def test_prompt_template_validates_input():
    """Test that invalid input raises validation error."""
    template = PromptTemplate(
        template_dir=Path("prompts"),
        version="v1",
        task="summarization",
    )
    
    with pytest.raises(ValueError):
        # Invalid style
        SummarizationInput(
            text="Test",
            style="invalid_style",
        )


def test_prompt_variables_are_present():
    """Test that all required variables are in template."""
    template_path = Path("prompts/v1/summarization/user.jinja2")
    template_content = template_path.read_text()
    
    required_vars = ["text", "max_words", "style"]
    for var in required_vars:
        assert f"{{{{ {var} }}}}" in template_content or f"{{% if {var} %}}" in template_content
```

### A/B Testing

```python
import random
from openai import AsyncOpenAI


class ABTestingPromptManager:
    """Manage A/B testing of prompt versions."""
    
    def __init__(self, registry: PromptRegistry):
        self.registry = registry
    
    async def get_prompt_version(self, task: str, user_id: str) -> PromptVersion:
        """
        Select prompt version based on A/B test configuration.
        
        Args:
            task: Task name
            user_id: User identifier for consistent assignment
            
        Returns:
            Selected prompt version
        """
        versions = self.registry.prompts.get(task, [])
        active_versions = [
            v for v in versions
            if v.status in ["production", "testing"] and v.traffic_percentage > 0
        ]
        
        if not active_versions:
            raise ValueError(f"No active versions for task: {task}")
        
        # Consistent assignment based on user_id hash
        user_hash = hash(user_id) % 100
        
        cumulative = 0.0
        for version in active_versions:
            cumulative += version.traffic_percentage
            if user_hash < cumulative:
                return version
        
        return active_versions[-1]  # Fallback
    
    async def log_result(
        self,
        task: str,
        version: str,
        user_id: str,
        input_data: dict,
        output: str,
        latency_ms: float,
        user_feedback: float | None = None,
    ) -> None:
        """Log A/B test result for analysis."""
        # In production, send to analytics system (e.g., Mixpanel, Amplitude)
        import json
        log_entry = {
            "task": task,
            "version": version,
            "user_id": user_id,
            "input": input_data,
            "output": output,
            "latency_ms": latency_ms,
            "user_feedback": user_feedback,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Append to log file (use proper analytics in production)
        log_path = Path(f"evaluations/{task}_ab_test.jsonl")
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")
```

### Regression Testing

```python
from pydantic import BaseModel
from typing import Any


class GoldenExample(BaseModel):
    """Golden dataset example for regression testing."""
    
    input: dict[str, Any]
    expected_output: str
    metadata: dict[str, Any] = {}


class RegressionTester:
    """Test prompts against golden dataset."""
    
    def __init__(
        self,
        golden_dataset: list[GoldenExample],
        similarity_threshold: float = 0.8,
    ):
        self.golden_dataset = golden_dataset
        self.similarity_threshold = similarity_threshold
    
    async def test_prompt_version(
        self,
        client: AsyncOpenAI,
        template: PromptTemplate,
    ) -> dict[str, float]:
        """
        Test prompt version against golden dataset.
        
        Args:
            client: OpenAI async client
            template: Prompt template to test
            
        Returns:
            Test results with metrics
        """
        from openai import AsyncOpenAI
        import asyncio
        
        results = []
        
        for example in self.golden_dataset:
            # Render prompt
            input_data = SummarizationInput(**example.input)
            prompts = template.render(input_data)
            
            # Generate output
            response = await client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
            )
            
            actual_output = response.choices[0].message.content or ""
            
            # Calculate similarity (simplified - use embeddings in production)
            similarity = self._calculate_similarity(
                actual_output,
                example.expected_output,
            )
            
            results.append({
                "input": example.input,
                "expected": example.expected_output,
                "actual": actual_output,
                "similarity": similarity,
                "passed": similarity >= self.similarity_threshold,
            })
        
        # Aggregate metrics
        passed = sum(1 for r in results if r["passed"])
        avg_similarity = sum(r["similarity"] for r in results) / len(results)
        
        return {
            "total_examples": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "avg_similarity": avg_similarity,
            "results": results,
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity (simplified)."""
        # In production, use embeddings:
        # embeddings = await client.embeddings.create(...)
        # similarity = cosine_similarity(emb1, emb2)
        
        # Simplified: Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
```

---

## Anti-Patrones de Seguridad

### ❌ Exposición de System Prompts
**Problema:** Permitir que el usuario consulte o modifique el system prompt mediante inyección.
**Solución:** Nunca incluir el system prompt completo en logs visibles por el usuario y usar guardrails para detectar intentos de "ignore previous instructions".

### ❌ Data Poisoning en Few-Shot Examples
**Problema:** Incluir ejemplos dinámicos provenientes de fuentes no confiables que pueden alterar el comportamiento del modelo.
**Solución:** Usar solo ejemplos "golden" validados manualmente o por un pipeline de confianza.

### ❌ Fuga de Contexto entre Sesiones
**Problema:** Reutilizar el mismo historial de chat para diferentes usuarios o inquilinos (tenants).
**Solución:** Limpiar rigurosamente el estado de la memoria al cambiar de contexto de usuario.

---

## Instructions for the Agent

1. **Prompts as Code**: NEVER hardcode prompts in application code. Store in files (`prompts/v1/task/`), version in Git, deploy through CI/CD.

2. **Template with Validation**: Use Jinja2 templates + Pydantic models:
   - Define input schema with Field descriptions
   - Validate variables before rendering
   - Fail fast on missing/invalid inputs

3. **Separate Concerns**: Keep distinct files for:
   - `system.txt` — System prompt
   - `user.jinja2` — User prompt template
   - `examples.json` — Few-shot examples
   - `metadata.json` — Version info, metrics, status

4. **Prompt Registry**: Maintain central registry (`registry.json`) with:
   - Version identifier (v1, v2, etc.)
   - Model and parameters (temperature, max_tokens)
   - Performance metrics (accuracy, latency, cost)
   - Deployment status (draft, testing, production, deprecated)
   - Traffic percentage for A/B testing

5. **Versioning Strategy**: Use semantic versioning:
   - v1, v2, v3 for major changes
   - Track in Git with meaningful commit messages
   - Keep previous versions deployable for rollback
   - Document what changed in metadata.json

6. **Testing Before Deploy**: ALWAYS test prompts before production:
   - Unit tests: Template renders correctly
   - Integration tests: LLM outputs meet quality thresholds
   - Regression tests: Golden dataset validation
   - A/B tests: Compare against baseline

7. **A/B Testing**: For production prompt changes:
   - Start with 10% traffic to new version
   - Monitor metrics (accuracy, latency, cost, user feedback)
   - Gradually increase if metrics improve
   - Rollback if metrics degrade

8. **Few-Shot Examples**: Quality > quantity:
   - 3-5 well-chosen examples are sufficient
   - Store separately from template
   - Validate examples manually
   - Update when task requirements change

9. **Cost Monitoring**: Track token usage per prompt version:
   - Log prompt length, completion length
   - Calculate cost per request
   - Alert on cost spikes
   - Optimize prompts to reduce tokens

10. **Chain-of-Thought (CoT)**: For reasoning tasks:
    - Include "Let's think step by step" or equivalent
    - Provide example reasoning in few-shot
    - Validate reasoning quality, not just final answer

11. **Prompt Patterns**: Choose appropriate pattern:
    - Zero-shot: Simple, well-defined tasks
    - Few-shot: Structured outputs, domain-specific
    - Chain-of-Thought: Math, logic, multi-step reasoning
    - ReAct: Agents with tool use
    - Self-Ask: Complex questions requiring decomposition

12. **Production Deployment**: Follow workflow:
    - Create prompt in `prompts/v{N}/task/`
    - Test with golden dataset
    - Register in registry.json
    - Deploy via CI/CD
    - Monitor metrics
    - A/B test before full rollout
    - Keep rollback plan ready

---

## Anti-Patterns

❌ **Hardcoding prompts in code** — Store in files, version in Git

❌ **No input validation** — Use Pydantic to validate template variables

❌ **Deploying without testing** — Always evaluate with metrics before production

❌ **Mixing business logic in prompts** — Keep prompts declarative, logic in code

❌ **No versioning** — Track prompt versions like code versions

❌ **Ignoring cost** — Monitor token usage, optimize prompt length

❌ **One-size-fits-all prompts** — Tailor prompts to specific models and tasks

❌ **No A/B testing** — Always compare new prompts against baseline

❌ **Forgetting examples in few-shot** — Quality > quantity (3-5 good examples)

❌ **Not logging prompt performance** — Track metrics in production for drift detection

❌ **Using outdated patterns** — Stay current with research (CoT, ReAct, etc.)

❌ **No rollback plan** — Keep previous versions deployable for quick rollback
