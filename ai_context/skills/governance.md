---
name: AI Governance
description: Frameworks for Responsible AI, compliance, and risk management.
---

# AI Governance

## Overview
Governance ensures that AI systems are safe, ethical, compliant, and reliable. It shifts the focus from "can we build it?" to "should we build it, and how do we control it?".

## Key Pillars

### 1. Responsible AI (RAI)
- **Fairness**: Testing for and mitigating bias in model outputs.
- **Transparency**: Clear disclosure of AI usage to users.
- **Accountability**: Human oversight for critical decisions.

### 2. Compliance & Privacy
- **GDPR / CCPA**: Data protection, right to be forgotten.
- **EU AI Act**: Risk-based classification of AI systems.
- **PII Protection**: Automatic detection and masking of sensitive data.

## Implementation Patterns

### Model Cards
Maintain a "Model Card" for every model in production, documenting:
- Intended use cases
- Limitations and known biases
- Training data sources
- Performance metrics

### PII Masking Pipeline
Sanitize inputs before they enter logs or model context.

```python
# Conceptual PII masking
text = "Call me at 555-0199"
sanitized = pii_analyzer.anonymize(text) # "Call me at <PHONE_NUMBER>"
```

### Audit Trails
Immutable logs of all AI interactions, decisions, and configuration changes.

## Best Practices
1.  **Human-in-the-Loop**: Mandatory for high-stakes domains (healthcare, finance).
2.  **Feedback Loops**: Mechanisms for users to report bad outputs, which feed into evaluation datasets.
3.  **Version Control for Prompts**: Treat prompts as code to track behavior changes over time.

## External Resources
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Microsoft Responsible AI Standard](https://query.prod.cms.rt.microsoft.com/cms/api/am/binary/RE5cmFl)
- [EU AI Act Explorer](https://artificialintelligenceact.eu/)
