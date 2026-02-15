---
name: AI Governance
description: Frameworks for Responsible AI, compliance, and risk management.
---

# Skill: AI Governance

> [!NOTE]
> Este documento define políticas de gobernanza que deben ser validadas por el DPO (Data Protection Officer) de la organización.

## Description

AI Governance ensures that AI systems are safe, ethical, compliant, and reliable throughout
their lifecycle. It encompasses privacy controls (PII masking, data minimization), regulatory
compliance (GDPR, EU AI Act), transparency mechanisms (model cards, explainability), and
operational oversight (audit trails, feedback loops). This skill provides concrete patterns
for building governable GenAI systems in production.

## Executive Summary

**Critical governance rules:**
- ALWAYS mask PII before logging — use Presidio or equivalent PII detection at boundary (applies to prompts, inputs, outputs)
- Immutable audit trails for ALL AI decisions — log timestamp, input hash, model version, prompt version, output summary, correlation ID
- Human-in-the-loop for high-stakes decisions — financial, medical, legal actions require approval gates
- Model Cards MANDATORY for all AI components — document limitations, bias evaluation, intended use, version history
- RAG responses MUST include source attribution — document_id, chunk_id, relevance_score for verification

**Read full skill when:** Implementing PII protection, ensuring GDPR/CCPA compliance, creating audit systems, performing bias evaluations, or building feedback loops for production AI systems.

## Deep Dive

---

> ⚠️ **DISCLAIMER LEGAL**: Los ejemplos de código en este documento son guías técnicas de implementación, NO constituyen asesoría legal. Antes de desplegar en producción sistemas que manejen PII o requieran compliance con GDPR, HIPAA, EU AI Act u otras regulaciones, consultar obligatoriamente con el equipo legal/compliance de la organización.

---

## Core Concepts

1. **Responsible AI (RAI)** — A framework of principles that guide AI development:
   fairness (testing for and mitigating bias), transparency (clear disclosure of AI usage),
   accountability (human oversight for critical decisions), and safety (preventing harmful
   outputs). RAI is not a checklist but an ongoing practice embedded in the development
   lifecycle.

2. **Privacy and PII Protection** — Automatic detection and masking of Personally
   Identifiable Information (PII) before data enters logs, model context, or storage.
   Uses tools like Microsoft Presidio for entity recognition (names, emails, phone
   numbers, SSNs) and configurable anonymization strategies (redaction, replacement,
   hashing).

3. **Regulatory Compliance** — Adherence to data protection laws (GDPR, CCPA) and
   AI-specific regulation (EU AI Act). Includes risk classification of AI systems,
   data subject rights (access, deletion, portability), and documentation requirements.
   Systems must be classifiable by risk level: unacceptable, high, limited, or minimal.

4. **Model Cards and Documentation** — Structured documentation for every model or
   AI component in production. Captures intended use cases, known limitations, training
   data sources, performance metrics, bias evaluation results, and version history.
   Model cards are living documents updated with each deployment.

5. **Audit Trails** — Immutable, structured logs of all AI decisions, configuration
   changes, prompt versions, and human overrides. Audit trails enable post-hoc analysis,
   regulatory reporting, and incident investigation. They must capture what decision was
   made, what inputs were used, which model version was active, and who approved it.

6. **Feedback Loops** — Mechanisms for end users, domain experts, and reviewers to
   report problematic outputs, flag bias, or rate quality. Feedback data feeds into
   evaluation datasets, retraining pipelines, and governance dashboards. Without
   feedback loops, governance is blind to production behavior.

## External Resources

### :shield: Frameworks & Standards
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
  *Best for:* Comprehensive risk identification, assessment, and mitigation for AI systems.
- [EU AI Act Explorer](https://artificialintelligenceact.eu/)
  *Best for:* Understanding risk-based classification and compliance requirements.
- [Microsoft Responsible AI Standard](https://query.prod.cms.rt.microsoft.com/cms/api/am/binary/RE5cmFl)
  *Best for:* Enterprise RAI governance framework with concrete implementation guidance.

### :mag: Explainability & Fairness
- [AI Fairness 360 (IBM)](https://aif360.readthedocs.io/)
  *Best for:* Bias detection and mitigation algorithms for classification models.
- [SHAP Documentation](https://shap.readthedocs.io/)
  *Best for:* Model-agnostic explanations using Shapley values.
- [LIME Documentation](https://lime-ml.readthedocs.io/)
  *Best for:* Local interpretable explanations for individual predictions.
- [Google PAIR (People + AI Research)](https://pair.withgoogle.com/)
  *Best for:* Human-centered AI design guidelines and research.

### :lock: Privacy & Security
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
  *Best for:* PII detection and anonymization in text and structured data.
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
  *Best for:* Security risks specific to LLM applications (prompt injection, data leakage).

### :book: Safety Research
- [Anthropic Safety Research](https://www.anthropic.com/research)
  *Best for:* Constitutional AI, RLHF safety, and alignment research.

## Instructions for the Agent

1. **Always mask PII before logging.** Never log raw user inputs, prompts, or model
   outputs that may contain PII. Use a PII detection pipeline (Presidio or equivalent)
   at the boundary before any logging or telemetry. This applies to structured logs,
   traces, and error messages.

2. **Maintain audit trails for all AI decisions.** Every LLM call, tool invocation,
   and agent decision must produce an immutable audit record with: timestamp, input
   hash (not raw PII), model version, prompt version, output summary, and correlation
   ID. Store audit records in append-only storage.

3. **Implement human-in-the-loop for high-stakes decisions.** Any AI-driven action
   with significant consequences (financial transactions, content moderation, medical
   triage) must include a human approval gate. The gate must have configurable timeout
   and fallback behavior.

4. **Version control all prompts.** Prompts are versioned artifacts tracked alongside
   code. Every prompt change must be reviewable, testable, and traceable to a specific
   deployment. Use prompt version IDs in audit trails.

5. **Document model limitations explicitly.** Maintain a Model Card for every model
   or AI component. Include known failure modes, bias evaluation results, and
   out-of-distribution behavior. Update on every significant model or prompt change.

6. **Implement feedback loops for production monitoring.** Provide mechanisms for users
   to report bad outputs (thumbs down, flag, detailed report). Feed this data into
   evaluation datasets and governance dashboards. No AI system should run in production
   without a feedback channel.

7. **Classify system risk level per EU AI Act.** At project inception, classify the
   system's risk level (unacceptable, high, limited, minimal). This classification
   determines required documentation, human oversight level, and compliance obligations.
   Document the classification rationale in architecture decision records.

## Code Examples

### PII Detection and Masking Pipeline with Presidio

```python
"""PII detection and masking using Microsoft Presidio."""
import logging
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)


@dataclass
class PIIMaskingResult:
    """Result of PII masking operation."""
    original_length: int
    masked_text: str
    entities_found: list[str]
    entity_count: int


class PIIMaskingPipeline:
    """Masks PII in text before logging or model input."""

    SUPPORTED_ENTITIES = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "CREDIT_CARD", "US_SSN", "IP_ADDRESS", "LOCATION",
    ]

    def __init__(self, language: str = "en"):
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._language = language
        self._operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "<SSN>"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        }

    def mask(self, text: str) -> PIIMaskingResult:
        """Detect and mask PII entities in the given text."""
        results: list[RecognizerResult] = self._analyzer.analyze(
            text=text,
            language=self._language,
            entities=self.SUPPORTED_ENTITIES,
        )

        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self._operators,
        )

        entities_found = [r.entity_type for r in results]

        if entities_found:
            logger.info(
                "pii_masked",
                extra={"entity_count": len(entities_found), "types": list(set(entities_found))},
            )

        return PIIMaskingResult(
            original_length=len(text),
            masked_text=anonymized.text,
            entities_found=entities_found,
            entity_count=len(entities_found),
        )


# Usage
pipeline = PIIMaskingPipeline()
result = pipeline.mask("Contact John Smith at john@example.com or 555-0199")
# result.masked_text -> "Contact <PERSON> at <EMAIL> or <PHONE>"
```

### Audit Trail Logging for AI Decisions

```python
"""Immutable audit trail for AI decisions and actions."""
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    LLM_CALL = "llm_call"
    TOOL_INVOCATION = "tool_invocation"
    AGENT_DECISION = "agent_decision"
    HUMAN_OVERRIDE = "human_override"
    PROMPT_CHANGE = "prompt_change"


@dataclass
class AuditRecord:
    """Single immutable audit entry."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = ""
    decision_type: DecisionType = DecisionType.LLM_CALL
    agent_id: str = ""
    model_version: str = ""
    prompt_version: str = ""
    input_hash: str = ""
    output_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "decision_type": self.decision_type.value,
            "agent_id": self.agent_id,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "input_hash": self.input_hash,
            "output_summary": self.output_summary,
            "metadata": self.metadata,
        }


class AuditTrailService:
    """Records and queries AI decision audit trails."""

    def __init__(self, storage_backend: Any = None):
        self._storage = storage_backend  # In production: append-only DB or log sink

    @staticmethod
    def hash_input(raw_input: str) -> str:
        """Hash input to avoid storing PII in audit records."""
        return hashlib.sha256(raw_input.encode()).hexdigest()[:16]

    def record_decision(
        self,
        decision_type: DecisionType,
        agent_id: str,
        raw_input: str,
        output_summary: str,
        model_version: str = "",
        prompt_version: str = "",
        correlation_id: str = "",
        metadata: dict | None = None,
    ) -> AuditRecord:
        """Create and persist an audit record."""
        record = AuditRecord(
            correlation_id=correlation_id or str(uuid.uuid4()),
            decision_type=decision_type,
            agent_id=agent_id,
            model_version=model_version,
            prompt_version=prompt_version,
            input_hash=self.hash_input(raw_input),
            output_summary=output_summary[:500],  # Truncate for storage
            metadata=metadata or {},
        )

        # Structured log as append-only audit entry
        logger.info("audit_record", extra=record.to_dict())

        # Persist to storage backend
        if self._storage:
            self._storage.append(record.to_dict())

        return record


# Usage
audit = AuditTrailService()
audit.record_decision(
    decision_type=DecisionType.LLM_CALL,
    agent_id="support-agent",
    raw_input="User asked about refund for order #12345",
    output_summary="Generated refund policy response",
    model_version="claude-3-sonnet-20240229",
    prompt_version="support-v2.3",
    correlation_id="req-abc-123",
)
```

### Model Card Dataclass with Versioning

```python
"""Structured Model Card for AI component documentation."""
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class BiasEvaluation:
    """Results of bias testing."""
    metric: str  # e.g., "demographic_parity", "equalized_odds"
    protected_attribute: str  # e.g., "gender", "ethnicity"
    score: float
    threshold: float
    passed: bool
    details: str = ""


@dataclass
class ModelCard:
    """Living documentation for an AI model or component."""

    # Identity
    name: str
    version: str
    last_updated: date
    owner: str

    # Purpose
    description: str
    intended_use_cases: list[str]
    out_of_scope_uses: list[str]

    # Technical details
    model_type: str  # e.g., "LLM", "classifier", "embedding"
    base_model: str  # e.g., "claude-3-sonnet", "bert-base"
    training_data_summary: str
    prompt_version: str = ""

    # Performance
    performance_metrics: dict[str, float] = field(default_factory=dict)
    evaluation_datasets: list[str] = field(default_factory=list)

    # Limitations
    known_limitations: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    # Bias & Fairness
    bias_evaluations: list[BiasEvaluation] = field(default_factory=list)

    # Risk
    eu_ai_act_risk_level: str = "minimal"  # unacceptable, high, limited, minimal
    risk_classification_rationale: str = ""

    # Changelog
    changelog: list[dict[str, Any]] = field(default_factory=list)

    def add_changelog_entry(self, version: str, description: str, author: str) -> None:
        self.changelog.append({
            "version": version,
            "date": date.today().isoformat(),
            "description": description,
            "author": author,
        })

    def bias_summary(self) -> dict[str, bool]:
        """Return pass/fail summary of all bias evaluations."""
        return {
            f"{b.protected_attribute}_{b.metric}": b.passed
            for b in self.bias_evaluations
        }


# Usage
card = ModelCard(
    name="Customer Support Agent",
    version="2.3.0",
    last_updated=date(2025, 1, 15),
    owner="ml-platform-team",
    description="Handles tier-1 customer support queries with RAG over knowledge base.",
    intended_use_cases=["Answering product questions", "Refund policy lookup"],
    out_of_scope_uses=["Medical advice", "Legal counsel", "Financial recommendations"],
    model_type="LLM",
    base_model="claude-3-sonnet-20240229",
    training_data_summary="Fine-tuned on 50k anonymized support tickets (2023-2024).",
    prompt_version="support-v2.3",
    performance_metrics={"answer_relevancy": 0.89, "faithfulness": 0.92},
    known_limitations=[
        "May hallucinate product features not in knowledge base",
        "Limited accuracy for multi-language queries",
    ],
    eu_ai_act_risk_level="limited",
    risk_classification_rationale="Customer support is a limited-risk application with transparency obligations.",
)
```

### Feedback Collection Endpoint

```python
"""Feedback collection for AI output quality monitoring."""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    FLAG_HARMFUL = "flag_harmful"
    FLAG_INACCURATE = "flag_inaccurate"
    FLAG_BIASED = "flag_biased"
    DETAILED_REPORT = "detailed_report"


@dataclass
class UserFeedback:
    """Structured user feedback on AI output."""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    feedback_type: FeedbackType = FeedbackType.THUMBS_DOWN
    user_id: str = ""
    comment: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackCollector:
    """Collects and routes user feedback on AI outputs."""

    def __init__(self, storage_backend: object = None):
        self._storage = storage_backend
        self._escalation_types = {
            FeedbackType.FLAG_HARMFUL,
            FeedbackType.FLAG_BIASED,
        }

    def submit(
        self,
        correlation_id: str,
        feedback_type: FeedbackType,
        user_id: str = "anonymous",
        comment: str = "",
    ) -> UserFeedback:
        """Record user feedback and escalate if needed."""
        feedback = UserFeedback(
            correlation_id=correlation_id,
            feedback_type=feedback_type,
            user_id=user_id,
            comment=comment[:2000],  # Limit comment length
        )

        logger.info("feedback_received", extra={
            "feedback_id": feedback.feedback_id,
            "correlation_id": correlation_id,
            "type": feedback_type.value,
        })

        if self._storage:
            self._storage.append(feedback)  # type: ignore[attr-defined]

        if feedback_type in self._escalation_types:
            self._escalate(feedback)

        return feedback

    def _escalate(self, feedback: UserFeedback) -> None:
        """Escalate harmful or biased output reports for human review."""
        logger.warning("feedback_escalated", extra={
            "feedback_id": feedback.feedback_id,
            "type": feedback.feedback_type.value,
            "correlation_id": feedback.correlation_id,
        })
        # In production: notify Slack, PagerDuty, or governance dashboard


# Usage
collector = FeedbackCollector()
collector.submit(
    correlation_id="req-abc-123",
    feedback_type=FeedbackType.FLAG_INACCURATE,
    user_id="user-456",
    comment="The response cited a product feature that does not exist.",
)
```

## Anti-Patterns to Avoid

### :x: Logging Raw Prompts with PII

**Problem:** Writing user inputs or model outputs directly to logs without sanitization
exposes PII to log aggregation systems, violating GDPR and creating data breach risk.

**Example:**
```python
# BAD: raw user input in logs
logger.info(f"User query: {user_message}")
logger.info(f"LLM response: {llm_output}")
```

**Solution:** Run all text through a PII masking pipeline before logging. Log input
hashes instead of raw content for audit purposes.

### :x: No Human Oversight for Critical Decisions

**Problem:** Deploying AI for high-stakes decisions (financial, medical, legal) without
a human approval gate. Automated errors can have irreversible consequences.

**Example:**
```python
# BAD: automated refund with no review
refund_amount = await llm_agent.calculate_refund(order)
await payment_service.process_refund(refund_amount)  # No human check
```

**Solution:** Implement a human-in-the-loop gate for decisions above a configurable
threshold. Queue decisions for review with timeout and fallback behavior.

### :x: Deploying Without Model Cards

**Problem:** Running AI components in production without documented limitations, bias
evaluations, or intended use cases. Teams cannot assess risk or debug failures.

**Example:**
```python
# BAD: model deployed with no documentation
model = load_model("customer-support-v3")
# No record of what changed, what was tested, or known failure modes
```

**Solution:** Maintain a ModelCard dataclass for every AI component. Require model card
updates as part of the deployment checklist. Block deployments without current cards.

### :x: No User Feedback Mechanism

**Problem:** AI systems in production with no way for users to report bad outputs.
Governance teams are blind to quality degradation, bias, and harmful content.

**Example:**
```python
# BAD: fire-and-forget response
return {"response": llm_output}  # No feedback channel
```

**Solution:** Include feedback endpoints (thumbs up/down, flag, detailed report) in
every AI-facing interface. Route escalations to human reviewers. Feed data into
evaluation datasets.

## Governance Checklist

### Privacy & PII
- [ ] PII detection pipeline integrated at input/output boundaries
- [ ] Raw PII never appears in logs, traces, or error messages
- [ ] Data retention policies defined and enforced
- [ ] Data subject access and deletion requests handled
- [ ] PII masking tested with representative data samples

### Compliance
- [ ] EU AI Act risk level classified and documented
- [ ] GDPR/CCPA data processing agreements in place
- [ ] Data processing register maintained
- [ ] Regular compliance audits scheduled
- [ ] Legal review completed for AI use cases

### Transparency
- [ ] Model Cards maintained for all AI components
- [ ] AI usage disclosed to end users where required
- [ ] Prompt versions tracked and linked to deployments
- [ ] Decision explanations available for high-stakes outputs
- [ ] Changelog maintained for model and prompt updates

### Fairness & Bias
- [ ] Bias evaluation performed before each deployment
- [ ] Protected attributes identified and tested
- [ ] Fairness metrics defined with pass/fail thresholds
- [ ] Remediation plan for detected bias documented
- [ ] Regular re-evaluation scheduled (quarterly minimum)

### Operational Governance
- [ ] Audit trail captures all AI decisions with correlation IDs
- [ ] User feedback mechanism deployed in all AI interfaces
- [ ] Escalation path defined for harmful or biased outputs
- [ ] Human-in-the-loop gates active for high-stakes decisions
- [ ] Governance dashboard with key metrics (feedback rates, escalations, bias scores)
- [ ] Incident response plan for AI-specific failures documented

## Consent Management

### Core Concepts

Consent management tracks **what data a user authorized** for what purpose. In GenAI
systems, this is critical because user data may flow into: model context, RAG indices,
fine-tuning datasets, logs, and analytics. Each use requires explicit, granular consent.

### Consent Record Model

```python
"""Consent management for AI data processing."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ConsentPurpose(Enum):
    """Granular purposes for data processing."""
    MODEL_CONTEXT = "model_context"          # Data sent to LLM as context
    RAG_INDEXING = "rag_indexing"             # Data indexed in vector store
    FINE_TUNING = "fine_tuning"              # Data used for model training
    ANALYTICS = "analytics"                   # Aggregated usage analytics
    FEEDBACK_TRAINING = "feedback_training"   # User feedback used to improve model
    LOGGING = "logging"                       # Data retained in operational logs


class ConsentStatus(Enum):
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"


@dataclass
class ConsentRecord:
    """Immutable record of user consent."""
    consent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    purpose: ConsentPurpose = ConsentPurpose.MODEL_CONTEXT
    status: ConsentStatus = ConsentStatus.DENIED
    granted_at: datetime | None = None
    withdrawn_at: datetime | None = None
    expires_at: datetime | None = None
    legal_basis: str = ""  # "consent", "legitimate_interest", "contract"
    data_categories: list[str] = field(default_factory=list)  # ["name", "email", "query_history"]
    metadata: dict = field(default_factory=dict)


class ConsentManager:
    """Manages user consent for AI data processing."""

    def __init__(self, storage: object = None):
        self._storage = storage
        self._consents: dict[str, list[ConsentRecord]] = {}

    def grant(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        data_categories: list[str],
        legal_basis: str = "consent",
        expires_at: datetime | None = None,
    ) -> ConsentRecord:
        record = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            legal_basis=legal_basis,
            data_categories=data_categories,
        )
        self._consents.setdefault(user_id, []).append(record)
        return record

    def withdraw(self, user_id: str, purpose: ConsentPurpose) -> ConsentRecord | None:
        """Withdraw consent — triggers downstream data deletion."""
        for record in reversed(self._consents.get(user_id, [])):
            if record.purpose == purpose and record.status == ConsentStatus.GRANTED:
                record.status = ConsentStatus.WITHDRAWN
                record.withdrawn_at = datetime.now(timezone.utc)
                return record
        return None

    def is_allowed(self, user_id: str, purpose: ConsentPurpose) -> bool:
        """Check if user has active consent for a specific purpose."""
        now = datetime.now(timezone.utc)
        for record in reversed(self._consents.get(user_id, [])):
            if record.purpose != purpose:
                continue
            if record.status == ConsentStatus.WITHDRAWN:
                return False
            if record.status == ConsentStatus.GRANTED:
                if record.expires_at and record.expires_at < now:
                    return False
                return True
        return False

    def get_user_consents(self, user_id: str) -> list[ConsentRecord]:
        """Return all consent records for GDPR data subject access request."""
        return self._consents.get(user_id, [])
```

### Instructions for the Agent

8. **Always check consent before processing user data.** Before indexing data into
   RAG, sending it to an LLM, or using it for analytics, verify the user has active
   consent for that specific purpose. Default to denied if no consent record exists.

---

## Data Subject Rights (GDPR / CCPA)

GenAI systems must support the full lifecycle of data subject rights. This is not
optional — it's a legal requirement in EU, California, and expanding globally.

### Right to Access, Deletion, and Portability

```python
"""GDPR data subject rights handler for GenAI systems."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RightType(Enum):
    ACCESS = "access"           # Right to know what data is held
    DELETION = "deletion"       # Right to erasure ("right to be forgotten")
    PORTABILITY = "portability"  # Right to receive data in machine-readable format
    RECTIFICATION = "rectification"  # Right to correct inaccurate data
    OBJECTION = "objection"     # Right to object to processing


class DataStore(Protocol):
    """Protocol for any store that holds user data."""
    def get_user_data(self, user_id: str) -> list[dict[str, Any]]: ...
    def delete_user_data(self, user_id: str) -> int: ...
    def export_user_data(self, user_id: str) -> dict[str, Any]: ...


@dataclass
class SubjectRightRequest:
    request_id: str
    user_id: str
    right_type: RightType
    requested_at: datetime
    completed_at: datetime | None = None
    status: str = "pending"  # pending, in_progress, completed, failed


class DataSubjectRightsHandler:
    """Orchestrates data subject right requests across all data stores.

    In a GenAI system, user data may exist in:
    - Relational DB (user profiles, preferences)
    - Vector store (RAG embeddings from user documents)
    - Log aggregation (operational logs)
    - LLM provider (if data was sent for fine-tuning)
    - Cache layers (Redis, semantic cache)
    """

    def __init__(self, stores: dict[str, DataStore]):
        self._stores = stores  # e.g. {"postgres": pg, "vector_store": qdrant, "cache": redis}

    async def handle_deletion(self, user_id: str) -> dict[str, int]:
        """Delete user data from ALL stores. GDPR Art. 17."""
        results = {}
        for store_name, store in self._stores.items():
            try:
                deleted_count = store.delete_user_data(user_id)
                results[store_name] = deleted_count
                logger.info(
                    "data_deletion_completed",
                    extra={"store": store_name, "user_id_hash": hash(user_id), "count": deleted_count},
                )
            except Exception:
                logger.exception("data_deletion_failed", extra={"store": store_name})
                results[store_name] = -1  # Signal failure
        return results

    async def handle_access(self, user_id: str) -> dict[str, Any]:
        """Return all user data from all stores. GDPR Art. 15."""
        data: dict[str, Any] = {}
        for store_name, store in self._stores.items():
            try:
                data[store_name] = store.export_user_data(user_id)
            except Exception:
                logger.exception("data_access_failed", extra={"store": store_name})
                data[store_name] = {"error": "retrieval_failed"}
        return data

    async def handle_portability(self, user_id: str) -> dict[str, Any]:
        """Export user data in machine-readable format. GDPR Art. 20."""
        data = await self.handle_access(user_id)
        return {
            "format": "json",
            "schema_version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
```

### Instructions for the Agent

9. **Implement data deletion across ALL stores.** When a user requests deletion,
   data must be removed from: relational DB, vector store (embeddings), cache,
   logs (within retention policy), and any LLM provider datasets. Partial deletion
   is a compliance violation.

10. **Respond to data subject requests within 30 days.** GDPR requires response
    within one calendar month. Track requests with status and timestamps. Automate
    as much as possible.

---

## Data Residency and Sovereignty

### Core Concepts

Data residency rules dictate **where** data can be stored and processed. This is
critical for GenAI because LLM API calls send data to external providers, often
across borders.

### Data Residency Configuration

```python
"""Data residency enforcement for GenAI API calls."""
from dataclasses import dataclass, field
from enum import Enum


class Region(Enum):
    EU = "eu"
    US = "us"
    APAC = "apac"
    LATAM = "latam"


@dataclass
class ResidencyPolicy:
    """Defines where data can be processed for a given tenant."""
    tenant_id: str
    allowed_regions: list[Region]
    allowed_providers: list[str]  # e.g. ["azure-eu", "anthropic-eu"]
    block_cross_border: bool = True


@dataclass
class LLMEndpoint:
    """An LLM provider endpoint with region metadata."""
    provider: str
    model: str
    region: Region
    endpoint_url: str
    supports_data_residency: bool = True


class ResidencyRouter:
    """Routes LLM calls to compliant endpoints based on tenant policy."""

    def __init__(self, endpoints: list[LLMEndpoint]):
        self._endpoints = endpoints
        self._by_region: dict[Region, list[LLMEndpoint]] = {}
        for ep in endpoints:
            self._by_region.setdefault(ep.region, []).append(ep)

    def get_endpoint(self, policy: ResidencyPolicy, preferred_model: str = "") -> LLMEndpoint:
        """Select a compliant endpoint for the tenant's residency policy."""
        for region in policy.allowed_regions:
            candidates = self._by_region.get(region, [])
            for ep in candidates:
                if ep.provider in policy.allowed_providers or not policy.allowed_providers:
                    if not preferred_model or ep.model == preferred_model:
                        return ep
            # Fallback: any endpoint in allowed region
            if candidates:
                return candidates[0]

        raise ValueError(
            f"No compliant endpoint for tenant={policy.tenant_id}, "
            f"allowed_regions={policy.allowed_regions}"
        )


# Usage
endpoints = [
    LLMEndpoint("azure", "gpt-4o", Region.EU, "https://eu.openai.azure.com/..."),
    LLMEndpoint("anthropic", "claude-sonnet", Region.US, "https://api.anthropic.com/..."),
    LLMEndpoint("anthropic", "claude-sonnet", Region.EU, "https://eu.api.anthropic.com/..."),
]
router = ResidencyRouter(endpoints)
policy = ResidencyPolicy(tenant_id="acme-eu", allowed_regions=[Region.EU], allowed_providers=[])
endpoint = router.get_endpoint(policy)
# Returns EU endpoint only
```

### Instructions for the Agent

11. **Never send data to a region not allowed by the tenant's residency policy.**
    Always route LLM calls through a residency-aware router. If no compliant
    endpoint exists, fail loudly — never silently fall back to a non-compliant region.

---

## Industry-Specific Compliance

### Healthcare (HIPAA)

```python
"""HIPAA-specific safeguards for healthcare GenAI systems."""
from dataclasses import dataclass
from enum import Enum


class PHICategory(Enum):
    """Protected Health Information categories under HIPAA."""
    PATIENT_NAME = "patient_name"
    MEDICAL_RECORD_NUMBER = "mrn"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    DATE_OF_SERVICE = "date_of_service"
    SSN = "ssn"
    INSURANCE_ID = "insurance_id"


@dataclass
class HIPAAConfig:
    """HIPAA compliance configuration for GenAI pipelines."""
    # BAA (Business Associate Agreement) must exist with LLM provider
    baa_verified: bool = False
    # Minimum necessary rule: only send PHI needed for the task
    minimum_necessary: bool = True
    # De-identification: remove PHI before sending to LLM when possible
    de_identify_before_llm: bool = True
    # Encryption requirements
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    # Audit every access to PHI
    audit_phi_access: bool = True
    # Allowed LLM providers (must have BAA)
    allowed_providers: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.allowed_providers is None:
            self.allowed_providers = []

    def validate(self) -> list[str]:
        """Return list of compliance violations."""
        violations = []
        if not self.baa_verified:
            violations.append("BAA not verified with LLM provider")
        if not self.encryption_at_rest:
            violations.append("Encryption at rest not enabled")
        if not self.encryption_in_transit:
            violations.append("Encryption in transit not enabled")
        if not self.audit_phi_access:
            violations.append("PHI access auditing not enabled")
        return violations


# Usage — validate before starting the pipeline
config = HIPAAConfig(baa_verified=True, allowed_providers=["azure-hipaa"])
violations = config.validate()
if violations:
    raise RuntimeError(f"HIPAA compliance violations: {violations}")
```

### Financial Services (PCI-DSS / SOX)

```python
"""Financial compliance safeguards for GenAI systems."""
from dataclasses import dataclass


@dataclass
class FinancialComplianceConfig:
    """PCI-DSS and SOX compliance configuration."""

    # PCI-DSS: Never send card data to LLM
    pci_entities_blocked: list[str] = None  # type: ignore[assignment]
    # SOX: All financial decisions by AI must be auditable
    sox_audit_required: bool = True
    # Segregation of duties: AI cannot both recommend and approve
    segregation_of_duties: bool = True
    # Data retention for financial audit (7 years SOX)
    retention_years: int = 7

    def __post_init__(self) -> None:
        if self.pci_entities_blocked is None:
            self.pci_entities_blocked = [
                "CREDIT_CARD", "IBAN", "US_BANK_NUMBER",
                "CRYPTO_WALLET", "US_SSN",
            ]

    def validate(self) -> list[str]:
        violations = []
        if not self.sox_audit_required:
            violations.append("SOX audit trail not enabled")
        if not self.segregation_of_duties:
            violations.append("Segregation of duties not enforced")
        if self.retention_years < 7:
            violations.append(f"Retention {self.retention_years}y < SOX minimum 7y")
        return violations
```

### Instructions for the Agent

12. **Apply industry-specific compliance before generic governance.** If the system
    handles healthcare data, enforce HIPAA (BAA, de-identification, minimum necessary).
    If financial data, enforce PCI-DSS (block card data from LLM) and SOX (7-year
    audit retention). Industry rules override general defaults.

---

## Data Retention and Lifecycle

### Retention Policy Enforcement

```python
"""Automated data retention enforcement for GenAI systems."""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Defines how long data is retained in a specific store."""
    store_name: str
    data_category: str  # "llm_logs", "user_queries", "embeddings", "audit_trails"
    retention_days: int
    deletion_strategy: str = "hard_delete"  # "hard_delete", "soft_delete", "anonymize"
    legal_hold: bool = False  # If True, skip deletion regardless of policy


class DataStore(Protocol):
    def delete_older_than(self, cutoff: datetime, category: str) -> int: ...
    def anonymize_older_than(self, cutoff: datetime, category: str) -> int: ...


class RetentionEnforcer:
    """Runs periodically to enforce data retention policies."""

    def __init__(self, policies: list[RetentionPolicy], stores: dict[str, DataStore]):
        self._policies = policies
        self._stores = stores

    def enforce(self) -> dict[str, Any]:
        """Execute retention policies across all stores."""
        now = datetime.now(timezone.utc)
        results = {}

        for policy in self._policies:
            if policy.legal_hold:
                logger.info("retention_skipped_legal_hold", extra={"store": policy.store_name})
                continue

            store = self._stores.get(policy.store_name)
            if not store:
                continue

            cutoff = now - timedelta(days=policy.retention_days)

            if policy.deletion_strategy == "anonymize":
                count = store.anonymize_older_than(cutoff, policy.data_category)
            else:
                count = store.delete_older_than(cutoff, policy.data_category)

            results[f"{policy.store_name}/{policy.data_category}"] = {
                "deleted": count,
                "cutoff": cutoff.isoformat(),
                "strategy": policy.deletion_strategy,
            }
            logger.info("retention_enforced", extra=results[f"{policy.store_name}/{policy.data_category}"])

        return results


# Standard retention policies for GenAI systems
STANDARD_POLICIES = [
    RetentionPolicy("logs", "llm_invocations", retention_days=90),
    RetentionPolicy("logs", "user_queries", retention_days=30, deletion_strategy="anonymize"),
    RetentionPolicy("cache", "semantic_cache", retention_days=7, deletion_strategy="hard_delete"),
    RetentionPolicy("vector_store", "user_embeddings", retention_days=365),
    RetentionPolicy("audit_db", "audit_trails", retention_days=2555, legal_hold=True),  # 7 years
]
```

---

## Content Safety and Output Filtering

### Output Guardrails

```python
"""Content safety guardrails for LLM outputs."""
import re
from dataclasses import dataclass
from enum import Enum


class SafetyCategory(Enum):
    HARMFUL_CONTENT = "harmful_content"
    PII_LEAKAGE = "pii_leakage"
    PROMPT_INJECTION = "prompt_injection"
    HALLUCINATED_REFERENCE = "hallucinated_reference"
    TOXIC_LANGUAGE = "toxic_language"


@dataclass
class SafetyCheckResult:
    passed: bool
    category: SafetyCategory | None = None
    detail: str = ""
    confidence: float = 0.0


class OutputSafetyGuard:
    """Validates LLM outputs before returning to users.

    Layered approach:
    1. Rule-based checks (fast, deterministic)
    2. Regex pattern matching (PII leakage, known bad patterns)
    3. Optional LLM-as-judge for nuanced content (slower, use for high-risk)
    """

    # Patterns that should never appear in LLM output
    BLOCKED_PATTERNS = [
        (r"sk-[a-zA-Z0-9]{20,}", SafetyCategory.PII_LEAKAGE, "API key detected in output"),
        (r"(?i)system\s*prompt\s*:", SafetyCategory.PROMPT_INJECTION, "System prompt leak attempt"),
        (r"\b\d{3}-\d{2}-\d{4}\b", SafetyCategory.PII_LEAKAGE, "SSN pattern in output"),
        (r"(?i)(ignore|disregard)\s+(previous|above)\s+(instructions?|prompt)", SafetyCategory.PROMPT_INJECTION, "Injection echo"),
    ]

    def check(self, output: str) -> list[SafetyCheckResult]:
        """Run all safety checks on LLM output."""
        failures = []

        for pattern, category, detail in self.BLOCKED_PATTERNS:
            if re.search(pattern, output):
                failures.append(SafetyCheckResult(
                    passed=False,
                    category=category,
                    detail=detail,
                    confidence=1.0,
                ))

        return failures

    def is_safe(self, output: str) -> bool:
        """Quick check — returns False if any safety check fails."""
        return len(self.check(output)) == 0

    def sanitize(self, output: str) -> str:
        """Remove known dangerous patterns from output."""
        sanitized = output
        for pattern, _, _ in self.BLOCKED_PATTERNS:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized)
        return sanitized


# Usage in LLM call boundary
guard = OutputSafetyGuard()
llm_output = "Here is your answer... sk-abc123def456ghi789jkl"  # Leaked API key
if not guard.is_safe(llm_output):
    llm_output = guard.sanitize(llm_output)
```

### Instructions for the Agent

13. **Always filter LLM outputs before returning to users.** Apply the output safety
    guard at the boundary layer. Never trust raw LLM output. Check for PII leakage,
    prompt injection echoes, and blocked patterns. Sanitize or reject unsafe outputs.

---

## Explainability for GenAI

### Source Attribution in RAG

```python
"""Explainability: source attribution for RAG responses."""
from dataclasses import dataclass, field


@dataclass
class SourceAttribution:
    """Links a response segment to its source document."""
    document_id: str
    document_title: str
    chunk_id: str
    relevance_score: float
    page_number: int | None = None
    url: str = ""


@dataclass
class ExplainableResponse:
    """LLM response with attribution and confidence."""
    answer: str
    sources: list[SourceAttribution] = field(default_factory=list)
    confidence_score: float = 0.0  # 0-1, based on retrieval quality
    reasoning_trace: list[str] = field(default_factory=list)  # Steps the agent took
    caveats: list[str] = field(default_factory=list)  # Known limitations for this answer

    @property
    def has_sources(self) -> bool:
        return len(self.sources) > 0

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence_score >= 0.8 and self.has_sources


# Usage
response = ExplainableResponse(
    answer="The refund policy allows returns within 30 days.",
    sources=[
        SourceAttribution(
            document_id="doc-123",
            document_title="Refund Policy v2.1",
            chunk_id="chunk-456",
            relevance_score=0.94,
            page_number=3,
            url="https://internal.docs/refund-policy",
        ),
    ],
    confidence_score=0.92,
    reasoning_trace=[
        "Retrieved 5 chunks from knowledge base",
        "Reranked by relevance, top chunk score=0.94",
        "Generated answer grounded in top 2 chunks",
    ],
    caveats=["Policy may have been updated after index date 2025-01-15"],
)
```

### Instructions for the Agent

14. **Always provide source attribution for RAG responses.** Every answer derived
    from retrieved documents must include: source document ID, relevance score,
    and optionally page/section. Users and auditors must be able to verify claims
    against source material.

15. **Include confidence scores and caveats.** When retrieval quality is low
    (relevance < 0.7), flag the response as low-confidence. When the knowledge
    base may be stale, include a caveat about the index date.

---

## Governance Decision Matrix

Use this matrix to determine the governance level required for a given system:

| Factor | Minimal | Limited | High | Unacceptable |
|--------|---------|---------|------|-------------|
| **PII handling** | No PII | Anonymized PII | Direct PII | Biometric + sensitive |
| **Decision impact** | Informational | Recommendations | Automated actions | Irreversible actions |
| **Industry** | General | E-commerce | Finance, Education | Healthcare, Legal, HR |
| **User type** | Internal tools | B2B | B2C | Vulnerable populations |
| **Required governance** | Audit logs | + Model cards, feedback | + HITL, bias testing, compliance | Prohibited or strict review |

### Required Governance by Level

**Minimal**: Audit trails, basic logging, PII masking in logs.

**Limited**: All minimal + Model Cards, user feedback mechanism, prompt versioning,
transparency disclosure.

**High**: All limited + Human-in-the-loop gates, bias evaluation, industry compliance
(HIPAA/PCI-DSS/SOX), data residency enforcement, consent management, regular audits.

**Unacceptable**: System must not be deployed, or requires exceptional approval with
full governance stack + external audit + legal review.

---

## Updated Anti-Patterns

### :x: Sending Data Cross-Border Without Residency Check

**Problem:** LLM API calls to US-based providers with EU user data violates GDPR
data transfer rules when no adequacy decision or SCCs are in place.

**Solution:** Use a ResidencyRouter that maps tenant policies to compliant endpoints.
Fail loudly if no compliant endpoint exists.

### :x: No Consent Tracking for RAG Indexing

**Problem:** Indexing user documents into a vector store without explicit consent
for that purpose. Users may consent to "chat" but not to their data being indexed.

**Solution:** Check `ConsentManager.is_allowed(user_id, ConsentPurpose.RAG_INDEXING)`
before adding any document to the vector store.

### :x: Incomplete Deletion on GDPR Erasure Request

**Problem:** Deleting user data from the main DB but forgetting vector store
embeddings, semantic cache entries, or log aggregation systems.

**Solution:** Use `DataSubjectRightsHandler` that orchestrates deletion across ALL
registered data stores. Test deletion completeness regularly.

### :x: No Confidence Signal on Low-Quality RAG Responses

**Problem:** Returning RAG answers with low retrieval scores as if they were
high-confidence. Users trust the output and make bad decisions.

**Solution:** Include `confidence_score` in every response. Flag low-confidence
answers visually. Add caveats when the knowledge base may be stale.

---

## Updated Governance Checklist

### Privacy & PII
- [ ] PII detection pipeline integrated at input/output boundaries
- [ ] Raw PII never appears in logs, traces, or error messages
- [ ] Data retention policies defined and enforced with automated cleanup
- [ ] Data subject access and deletion requests handled across ALL stores
- [ ] PII masking tested with representative data samples
- [ ] Consent management system tracks per-purpose user authorization

### Compliance
- [ ] EU AI Act risk level classified and documented
- [ ] GDPR/CCPA data processing agreements in place
- [ ] Data processing register maintained
- [ ] Regular compliance audits scheduled
- [ ] Legal review completed for AI use cases
- [ ] Industry-specific compliance validated (HIPAA/PCI-DSS/SOX if applicable)
- [ ] Data residency policies enforced with region-aware routing
- [ ] Consent records exportable for regulatory audit

### Transparency & Explainability
- [ ] Model Cards maintained for all AI components
- [ ] AI usage disclosed to end users where required
- [ ] Prompt versions tracked and linked to deployments
- [ ] Decision explanations available for high-stakes outputs
- [ ] Changelog maintained for model and prompt updates
- [ ] Source attribution provided for RAG-based answers
- [ ] Confidence scores included in responses

### Fairness & Bias
- [ ] Bias evaluation performed before each deployment
- [ ] Protected attributes identified and tested
- [ ] Fairness metrics defined with pass/fail thresholds
- [ ] Remediation plan for detected bias documented
- [ ] Regular re-evaluation scheduled (quarterly minimum)

### Content Safety
- [ ] Output safety guard active at all LLM response boundaries
- [ ] Blocked patterns list maintained (API keys, SSNs, injection echoes)
- [ ] Sanitization applied before returning flagged outputs
- [ ] LLM-as-judge configured for high-risk content categories

### Operational Governance
- [ ] Audit trail captures all AI decisions with correlation IDs
- [ ] User feedback mechanism deployed in all AI interfaces
- [ ] Escalation path defined for harmful or biased outputs
- [ ] Human-in-the-loop gates active for high-stakes decisions
- [ ] Governance dashboard with key metrics (feedback rates, escalations, bias scores)
- [ ] Incident response plan for AI-specific failures documented
- [ ] Retention enforcer runs on schedule with documented results

## Additional References

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
  — Comprehensive risk management guidance for AI systems.
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — Open-source PII
  detection and anonymization toolkit.
- [EU AI Act Explorer](https://artificialintelligenceact.eu/) — Interactive guide
  to EU AI Act requirements and risk classification.
- [AI Fairness 360](https://aif360.readthedocs.io/) — IBM toolkit for detecting
  and mitigating bias in machine learning models.
- [Anthropic Safety Research](https://www.anthropic.com/research) — Research on
  AI alignment, Constitutional AI, and safety evaluations.
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
  — US healthcare data protection requirements.
- [PCI-DSS Requirements](https://www.pcisecuritystandards.org/)
  — Payment card data security standards.
- [GDPR Data Subject Rights](https://gdpr-info.eu/chapter-3/)
  — Full text of GDPR Chapter 3: Rights of the data subject.
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
  — NVIDIA's toolkit for adding programmable guardrails to LLM applications.
- [Guardrails AI](https://www.guardrailsai.com/)
  — Output validation framework for LLM responses.
