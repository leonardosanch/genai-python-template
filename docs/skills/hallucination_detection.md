# Skill: Hallucination Detection & Mitigation

## Description

Comprehensive strategies and techniques for detecting, measuring, and mitigating hallucinations in LLM outputs. This skill covers research-backed methods, production-ready tools, and integration patterns for building reliable GenAI systems.

## Executive Summary

**Critical rules (always enforce):**
- **Never trust LLM outputs blindly** — always validate against ground truth when available
- **Measure faithfulness systematically** — use automated metrics (faithfulness score ≥ 0.7 for production)
- **Implement citation verification** — require LLMs to cite sources and validate those citations
- **Use self-consistency checks** — generate multiple outputs and flag disagreements
- **Monitor hallucination rate in production** — alert when rate exceeds threshold (e.g., >5%)
- **Apply chain-of-verification (CoVe)** — have LLM verify its own claims before final output
- **Escalate to human review** — when confidence is low or stakes are high

**Read full skill when:** Building RAG systems, implementing fact-checking agents, deploying customer-facing LLM applications, evaluating LLM outputs, or debugging quality issues in production.

---

---

## Versiones y Thresholds de Calidad

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| ragas | >= 0.1.0 | Framework de evaluación |
| deepeval | >= 0.18.0 | Unit tests para LLMs |
| langchain | >= 0.1.0 | Orquestación base |

### Acceptance Thresholds

| Métrica | Target | Acción si < Target |
|---------|--------|-------------------|
| Faithfulness | >= 0.7 | Rechazar respuesta |
| Answer Relevancy | >= 0.8 | Re-intentar prompt |
| Context Precision | >= 0.7 | Revisar retrieval |

---

## Deep Dive

## Core Concepts

1. **Faithfulness vs Factuality**
   - **Faithfulness**: LLM output is grounded in provided context (RAG scenario)
   - **Factuality**: LLM output matches real-world truth (knowledge-based scenario)
   - Different metrics and approaches for each

2. **Intrinsic vs Extrinsic Hallucinations**
   - **Intrinsic**: Output contradicts the source/context
   - **Extrinsic**: Output adds information not present in source (may or may not be factual)
   - Both require different detection strategies

3. **Uncertainty Quantification**
   - LLMs can be overconfident even when wrong
   - Semantic entropy measures disagreement across multiple generations
   - Low entropy + wrong answer = dangerous hallucination

4. **Attribution and Provenance**
   - Require LLMs to cite sources for claims
   - Verify citations actually support the claims
   - Track information flow from retrieval → generation → output

5. **Multi-Stage Verification**
   - Generate → Verify → Refine pipeline
   - Self-correction through iterative prompting
   - External fact-checking agents

---

## External Resources

### 📚 Research Papers

- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)** (Manakul et al., 2023)  
  *Best for:* Zero-resource hallucination detection using sampling-based consistency
  
- **[Chain-of-Verification (CoVe)](https://arxiv.org/abs/2309.11495)** (Dhuliawala et al., 2023)  
  *Best for:* Reducing hallucinations through systematic self-verification
  
- **[Semantic Entropy](https://arxiv.org/abs/2302.09664)** (Kuhn et al., 2023)  
  *Best for:* Uncertainty estimation in LLM outputs via meaning-based clustering
  
- **[FActScore](https://arxiv.org/abs/2305.14251)** (Min et al., 2023)  
  *Best for:* Fine-grained factuality evaluation by decomposing into atomic facts
  
- **[TruthfulQA](https://arxiv.org/abs/2109.07958)** (Lin et al., 2021)  
  *Best for:* Benchmark for measuring truthfulness in question answering

### 🛠️ Tools & Frameworks

- **[DeepEval](https://docs.confident-ai.com/)** — Hallucination and faithfulness metrics  
  *Best for:* Production evaluation with pytest integration
  
- **[RAGAS](https://docs.ragas.io/)** — RAG-specific faithfulness metrics  
  *Best for:* End-to-end RAG pipeline evaluation
  
- **[LangSmith](https://docs.smith.langchain.com/)** — LLM tracing and evaluation  
  *Best for:* Debugging and monitoring LLM applications
  
- **[TruLens](https://www.trulens.org/)** — Feedback functions for LLM evaluation  
  *Best for:* Custom evaluation metrics and guardrails
  
- **[Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model)** — Hughes Hallucination Evaluation Model  
  *Best for:* Fast, specialized hallucination detection

### 📊 Evaluation Metrics

- **Faithfulness Score**: Measures if output is grounded in context (0-1)
- **Hallucination Rate**: Percentage of outputs containing hallucinations
- **Citation Accuracy**: Percentage of citations that correctly support claims
- **Factual Consistency**: Agreement between output and verified facts
- **Semantic Entropy**: Uncertainty measure based on semantic clustering

---

## Decision Tree: Hallucination Detection Strategy

```
START: Do you have ground truth context (RAG scenario)?
│
├─ YES (RAG/Grounded Generation)
│  │
│  ├─ Use FAITHFULNESS metrics
│  │  ├─ RAGAS faithfulness (context → answer)
│  │  ├─ DeepEval FaithfulnessMetric
│  │  └─ Citation verification (require sources)
│  │
│  └─ Threshold: faithfulness ≥ 0.7 for production
│     ├─ < 0.5: REJECT or flag for human review
│     ├─ 0.5-0.7: WARNING, consider regeneration
│     └─ ≥ 0.7: ACCEPT
│
└─ NO (Open-ended generation)
   │
   ├─ Use FACTUALITY metrics
   │  ├─ Self-consistency (sample N times, check agreement)
   │  ├─ Semantic entropy (measure uncertainty)
   │  ├─ External fact-checking (Wikipedia, knowledge base)
   │  └─ FActScore (decompose into atomic facts)
   │
   └─ Decision based on risk tolerance
      ├─ High stakes (medical, legal): Human review required
      ├─ Medium stakes: Self-consistency + CoVe
      └─ Low stakes: Monitor hallucination rate, sample audits

ALWAYS: Log outputs for drift detection and continuous evaluation
```

---

## Techniques Deep Dive

### 1. Self-Consistency

**Concept:** Generate multiple outputs for the same input and measure agreement. High disagreement indicates uncertainty and potential hallucination.

**Implementation:**

```python
from typing import Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import asyncio


class ConsistencyResult(BaseModel):
    """Result of self-consistency check."""
    
    responses: list[str] = Field(description="All generated responses")
    agreement_score: float = Field(
        ge=0.0, le=1.0, description="Semantic agreement score"
    )
    is_consistent: bool = Field(description="Whether responses are consistent")
    majority_answer: str | None = Field(description="Most common response")


async def self_consistency_check(
    client: AsyncOpenAI,
    prompt: str,
    n_samples: int = 5,
    threshold: float = 0.7,
) -> ConsistencyResult:
    """
    Generate multiple responses and check for consistency.
    
    Args:
        client: OpenAI async client
        prompt: Input prompt
        n_samples: Number of samples to generate
        threshold: Agreement threshold for consistency
        
    Returns:
        ConsistencyResult with agreement metrics
    """
    # Generate N independent responses
    tasks = [
        client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temp to get diversity
        )
        for _ in range(n_samples)
    ]
    
    completions = await asyncio.gather(*tasks)
    responses = [c.choices[0].message.content or "" for c in completions]
    
    # Calculate semantic similarity (simplified - use embeddings in production)
    # Here we use exact match for demonstration
    from collections import Counter
    
    response_counts = Counter(responses)
    majority_answer, majority_count = response_counts.most_common(1)[0]
    agreement_score = majority_count / n_samples
    
    return ConsistencyResult(
        responses=responses,
        agreement_score=agreement_score,
        is_consistent=agreement_score >= threshold,
        majority_answer=majority_answer if agreement_score >= threshold else None,
    )


# Usage
async def main() -> None:
    client = AsyncOpenAI()
    
    result = await self_consistency_check(
        client,
        prompt="What is the capital of France?",
        n_samples=5,
        threshold=0.7,
    )
    
    if not result.is_consistent:
        print(f"⚠️ Low consistency: {result.agreement_score:.2f}")
        print("Responses:", result.responses)
    else:
        print(f"✓ Consistent answer: {result.majority_answer}")
```

---

### 2. Citation Verification

**Concept:** Require LLM to cite sources for claims, then verify those citations actually support the claims.

**Implementation:**

```python
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single claim with citation."""
    
    text: str = Field(description="The claim text")
    citation: str = Field(description="Source citation (e.g., [1], [Doc 3])")


class VerifiedResponse(BaseModel):
    """LLM response with verified claims."""
    
    claims: list[Claim] = Field(description="All claims with citations")
    verified_claims: list[bool] = Field(description="Verification status per claim")
    overall_accuracy: float = Field(
        ge=0.0, le=1.0, description="Percentage of verified claims"
    )


async def verify_citations(
    client: AsyncOpenAI,
    claims: list[Claim],
    context_docs: dict[str, str],
) -> VerifiedResponse:
    """
    Verify that citations actually support the claims.
    
    Args:
        client: OpenAI async client
        claims: List of claims with citations
        context_docs: Mapping of citation ID to document text
        
    Returns:
        VerifiedResponse with verification results
    """
    verified = []
    
    for claim in claims:
        # Extract citation ID (e.g., "[1]" -> "1")
        citation_id = claim.citation.strip("[]")
        
        if citation_id not in context_docs:
            verified.append(False)
            continue
        
        # Ask LLM to verify if citation supports claim
        verification_prompt = f"""
Does the following source text support the claim?

Claim: {claim.text}

Source: {context_docs[citation_id]}

Answer with ONLY "yes" or "no".
"""
        
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.0,
        )
        
        answer = (response.choices[0].message.content or "").strip().lower()
        verified.append(answer == "yes")
    
    accuracy = sum(verified) / len(verified) if verified else 0.0
    
    return VerifiedResponse(
        claims=claims,
        verified_claims=verified,
        overall_accuracy=accuracy,
    )
```

---

### 3. Semantic Entropy

**Concept:** Measure uncertainty by generating multiple responses and clustering them semantically. High entropy = high uncertainty = potential hallucination.

**Implementation:**

```python
import numpy as np
from sklearn.cluster import DBSCAN
from openai import AsyncOpenAI


async def calculate_semantic_entropy(
    client: AsyncOpenAI,
    prompt: str,
    n_samples: int = 10,
) -> float:
    """
    Calculate semantic entropy of LLM responses.
    
    Higher entropy = more uncertainty = higher hallucination risk.
    
    Args:
        client: OpenAI async client
        prompt: Input prompt
        n_samples: Number of samples to generate
        
    Returns:
        Entropy value (0 = certain, higher = uncertain)
    """
    # Generate N responses
    tasks = [
        client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        for _ in range(n_samples)
    ]
    
    completions = await asyncio.gather(*tasks)
    responses = [c.choices[0].message.content or "" for c in completions]
    
    # Get embeddings for all responses
    embeddings_response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=responses,
    )
    
    embeddings = np.array([e.embedding for e in embeddings_response.data])
    
    # Cluster semantically similar responses
    clustering = DBSCAN(eps=0.3, min_samples=2, metric="cosine").fit(embeddings)
    labels = clustering.labels_
    
    # Calculate entropy based on cluster distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    return float(entropy)


# Usage
async def main() -> None:
    client = AsyncOpenAI()
    
    entropy = await calculate_semantic_entropy(
        client,
        prompt="Explain quantum entanglement in simple terms.",
        n_samples=10,
    )
    
    if entropy > 1.5:
        print(f"⚠️ High uncertainty (entropy={entropy:.2f}) - potential hallucination")
    else:
        print(f"✓ Low uncertainty (entropy={entropy:.2f})")
```

---

### 4. Factual Grounding

**Concept:** Ground LLM outputs in retrieved context and measure how well the output is supported by that context.

**Implementation:**

```python
from pydantic import BaseModel, Field


class GroundedResponse(BaseModel):
    """Response with grounding analysis."""
    
    answer: str = Field(description="LLM generated answer")
    context_used: list[str] = Field(description="Context documents used")
    grounding_score: float = Field(
        ge=0.0, le=1.0, description="How well answer is grounded in context"
    )
    unsupported_claims: list[str] = Field(
        description="Claims not supported by context"
    )


async def evaluate_grounding(
    client: AsyncOpenAI,
    answer: str,
    context: list[str],
) -> GroundedResponse:
    """
    Evaluate how well an answer is grounded in provided context.
    
    Args:
        client: OpenAI async client
        answer: LLM generated answer
        context: Context documents
        
    Returns:
        GroundedResponse with grounding analysis
    """
    # Use LLM to identify claims in the answer
    claims_prompt = f"""
Extract all factual claims from this answer as a JSON array of strings:

Answer: {answer}

Return ONLY a JSON array, nothing else.
"""
    
    claims_response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": claims_prompt}],
        response_format={"type": "json_object"},
    )
    
    import json
    claims = json.loads(claims_response.choices[0].message.content or "[]")
    
    # Check each claim against context
    unsupported = []
    for claim in claims:
        verification_prompt = f"""
Is this claim supported by ANY of the following context documents?

Claim: {claim}

Context:
{chr(10).join(f"[{i}] {doc}" for i, doc in enumerate(context))}

Answer with ONLY "yes" or "no".
"""
        
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.0,
        )
        
        if (response.choices[0].message.content or "").strip().lower() != "yes":
            unsupported.append(claim)
    
    grounding_score = 1.0 - (len(unsupported) / len(claims)) if claims else 1.0
    
    return GroundedResponse(
        answer=answer,
        context_used=context,
        grounding_score=grounding_score,
        unsupported_claims=unsupported,
    )
```

---

### 5. Chain-of-Verification (CoVe)

**Concept:** Generate answer → Generate verification questions → Answer verification questions → Refine original answer.

**Implementation:**

```python
from pydantic import BaseModel, Field


class VerificationStep(BaseModel):
    """Single verification question and answer."""
    
    question: str = Field(description="Verification question")
    answer: str = Field(description="Answer to verification question")
    contradicts_original: bool = Field(
        description="Whether this contradicts the original answer"
    )


class CoVeResult(BaseModel):
    """Result of Chain-of-Verification process."""
    
    original_answer: str = Field(description="Initial LLM answer")
    verification_steps: list[VerificationStep] = Field(
        description="Verification Q&A pairs"
    )
    refined_answer: str = Field(description="Final refined answer")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in final answer"
    )


async def chain_of_verification(
    client: AsyncOpenAI,
    question: str,
) -> CoVeResult:
    """
    Apply Chain-of-Verification to reduce hallucinations.
    
    Args:
        client: OpenAI async client
        question: User question
        
    Returns:
        CoVeResult with verification process
    """
    # Step 1: Generate initial answer
    initial_response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": question}],
    )
    original_answer = initial_response.choices[0].message.content or ""
    
    # Step 2: Generate verification questions
    verification_prompt = f"""
Given this question and answer, generate 3 verification questions to check for factual accuracy:

Question: {question}
Answer: {original_answer}

Return ONLY a JSON array of verification questions.
"""
    
    verification_response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": verification_prompt}],
        response_format={"type": "json_object"},
    )
    
    import json
    verification_questions = json.loads(
        verification_response.choices[0].message.content or "[]"
    )
    
    # Step 3: Answer verification questions independently
    verification_steps = []
    for vq in verification_questions:
        vq_response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": vq}],
        )
        vq_answer = vq_response.choices[0].message.content or ""
        
        # Check if verification answer contradicts original
        contradiction_prompt = f"""
Does this verification answer contradict the original answer?

Original: {original_answer}
Verification: {vq_answer}

Answer ONLY "yes" or "no".
"""
        
        contradiction_response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": contradiction_prompt}],
            temperature=0.0,
        )
        
        contradicts = (
            contradiction_response.choices[0].message.content or ""
        ).strip().lower() == "yes"
        
        verification_steps.append(
            VerificationStep(
                question=vq,
                answer=vq_answer,
                contradicts_original=contradicts,
            )
        )
    
    # Step 4: Refine answer based on verification
    contradictions = [v for v in verification_steps if v.contradicts_original]
    
    if contradictions:
        refine_prompt = f"""
The original answer has contradictions. Refine it based on verification:

Original: {original_answer}

Contradictions found:
{chr(10).join(f"- {v.question}: {v.answer}" for v in contradictions)}

Provide a refined, accurate answer.
"""
        
        refined_response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": refine_prompt}],
        )
        refined_answer = refined_response.choices[0].message.content or ""
        confidence = "medium" if len(contradictions) > 1 else "high"
    else:
        refined_answer = original_answer
        confidence = "high"
    
    return CoVeResult(
        original_answer=original_answer,
        verification_steps=verification_steps,
        refined_answer=refined_answer,
        confidence=confidence,
    )
```

---

## Integration Patterns

### RAG + Hallucination Detection

```python
from pydantic import BaseModel, Field


class RAGWithVerification(BaseModel):
    """RAG pipeline with built-in hallucination detection."""
    
    query: str
    retrieved_docs: list[str]
    answer: str
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    citations: list[str]
    is_trustworthy: bool


async def rag_with_verification(
    client: AsyncOpenAI,
    query: str,
    retrieved_docs: list[str],
    faithfulness_threshold: float = 0.7,
) -> RAGWithVerification:
    """
    RAG pipeline with automatic faithfulness verification.
    
    Args:
        client: OpenAI async client
        query: User query
        retrieved_docs: Retrieved context documents
        faithfulness_threshold: Minimum faithfulness score
        
    Returns:
        RAGWithVerification result
    """
    # Generate answer with citations
    context = "\n\n".join(
        f"[{i}] {doc}" for i, doc in enumerate(retrieved_docs)
    )
    
    prompt = f"""
Answer the question based ONLY on the provided context. Cite sources using [0], [1], etc.

Context:
{context}

Question: {query}

Answer:
"""
    
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    
    answer = response.choices[0].message.content or ""
    
    # Extract citations
    import re
    citations = re.findall(r"\[(\d+)\]", answer)
    
    # Evaluate faithfulness using grounding check
    grounding_result = await evaluate_grounding(client, answer, retrieved_docs)
    
    return RAGWithVerification(
        query=query,
        retrieved_docs=retrieved_docs,
        answer=answer,
        faithfulness_score=grounding_result.grounding_score,
        citations=citations,
        is_trustworthy=grounding_result.grounding_score >= faithfulness_threshold,
    )
```

### Multi-Agent Fact-Checking

```python
async def multi_agent_fact_check(
    client: AsyncOpenAI,
    claim: str,
) -> dict[str, bool | float]:
    """
    Use multiple verification agents to fact-check a claim.
    
    Args:
        client: OpenAI async client
        claim: Claim to verify
        
    Returns:
        Aggregated verification results
    """
    # Agent 1: Self-consistency
    consistency_result = await self_consistency_check(client, claim, n_samples=5)
    
    # Agent 2: Semantic entropy
    entropy = await calculate_semantic_entropy(client, claim, n_samples=10)
    
    # Agent 3: Chain-of-verification
    cove_result = await chain_of_verification(client, f"Is this true: {claim}")
    
    # Aggregate results
    consistency_vote = consistency_result.is_consistent
    entropy_vote = entropy < 1.0  # Low entropy = consistent
    cove_vote = cove_result.confidence in ["high", "medium"]
    
    votes = [consistency_vote, entropy_vote, cove_vote]
    confidence = sum(votes) / len(votes)
    
    return {
        "claim": claim,
        "is_verified": confidence >= 0.67,  # 2 out of 3 agents agree
        "confidence": confidence,
        "consistency_score": consistency_result.agreement_score,
        "entropy": entropy,
        "cove_confidence": cove_result.confidence,
    }
```

---

## Metrics & Evaluation

### Key Metrics

- **Faithfulness Score**: Percentage of answer grounded in context (0-1)
- **Factual Consistency**: Agreement with verified facts (0-1)
- **Hallucination Rate**: Percentage of outputs containing hallucinations
- **Citation Accuracy**: Percentage of valid citations

### Evaluation Code Example

```python
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate


async def evaluate_hallucination(
    query: str,
    answer: str,
    context: list[str],
) -> dict[str, float]:
    """
    Evaluate hallucination using DeepEval metrics.
    
    Args:
        query: User query
        answer: LLM generated answer
        context: Context documents
        
    Returns:
        Evaluation metrics
    """
    # Create test case
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context,
    )
    
    # Define metrics
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    hallucination_metric = HallucinationMetric(threshold=0.5)
    
    # Evaluate
    results = evaluate(
        test_cases=[test_case],
        metrics=[faithfulness_metric, hallucination_metric],
    )
    
    return {
        "faithfulness_score": faithfulness_metric.score,
        "hallucination_score": hallucination_metric.score,
        "passed": results.test_results[0].success,
    }
```

---

## Instructions for the Agent

1. **Never Trust Blindly**: ALWAYS validate LLM outputs, especially for high-stakes applications (medical, legal, financial). Implement verification before returning results to users.

2. **Measure Faithfulness**: For RAG scenarios, use faithfulness metrics (RAGAS, DeepEval). Set threshold ≥ 0.7 for production. Reject or flag outputs below 0.5 for human review.

3. **Require Citations**: In RAG systems, ALWAYS require LLM to cite sources using [1], [2] format. Verify citations actually support the claims before accepting output.

4. **Self-Consistency Checks**: For critical outputs, generate N samples (N=5-10) and measure agreement. High disagreement indicates uncertainty and potential hallucination.

5. **Chain-of-Verification (CoVe)**: For complex reasoning:
   - Generate initial answer
   - Generate verification questions
   - Answer verification questions independently
   - Refine answer based on verification results

6. **Semantic Entropy**: Calculate uncertainty by clustering multiple generations semantically. High entropy (>1.5) indicates high hallucination risk.

7. **Production Monitoring**: Track hallucination rate continuously:
   - Log all outputs with faithfulness scores
   - Alert when hallucination rate > 5%
   - Sample audit outputs regularly
   - Detect distribution shift in query patterns

8. **Escalation to Human**: When confidence is low (<0.7) or stakes are high, escalate to human review. Never auto-approve low-confidence outputs in critical domains.

9. **Multi-Agent Fact-Checking**: For important claims, use multiple verification agents:
   - Self-consistency agent
   - Semantic entropy agent
   - Chain-of-verification agent
   - Aggregate results (2 out of 3 agreement)

10. **Grounding Evaluation**: For every RAG output:
    - Extract factual claims
    - Verify each claim against context
    - Calculate grounding score
    - Flag unsupported claims
    - Require score ≥ 0.7 for production

---

## Anti-Patterns

❌ **Trusting LLM outputs without verification** — Always validate, especially for high-stakes applications

❌ **Using only exact string matching for consistency** — Use semantic similarity (embeddings) instead

❌ **Ignoring low-confidence outputs** — Flag and escalate to human review

❌ **Not logging hallucinations** — Track hallucination rate over time for drift detection

❌ **Skipping citation requirements** — Always require sources in RAG scenarios

❌ **Using single-shot generation for critical tasks** — Use self-consistency or CoVe

❌ **Not setting faithfulness thresholds** — Define clear acceptance criteria (e.g., ≥0.7)

❌ **Mixing hallucination detection with generation** — Separate concerns: generate, then verify

❌ **Not monitoring production hallucination rates** — Implement continuous evaluation

❌ **Assuming higher temperature = more hallucinations** — Hallucinations occur at all temperatures
