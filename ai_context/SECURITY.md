# Seguridad

## OWASP LLM Top 10 (2025)

| # | Riesgo | Mitigación |
|---|--------|------------|
| LLM01 | Prompt Injection | Input sanitization, system/user prompt separation, guardrails |
| LLM02 | Sensitive Information Disclosure | Output filtering, PII detection, data classification |
| LLM03 | Supply Chain Vulnerabilities | Dependency scanning, model provenance, SBOM |
| LLM04 | Data and Model Poisoning | Training data validation, fine-tune monitoring |
| LLM05 | Improper Output Handling | Output validation, sandboxed execution, Pydantic schemas |
| LLM06 | Excessive Agency | Least privilege, tool allowlisting, human-in-the-loop |
| LLM07 | System Prompt Leakage | Never expose system prompts, input/output filtering |
| LLM08 | Vector and Embedding Weaknesses | Access controls en vector stores, input validation |
| LLM09 | Misinformation | Grounding con RAG, fact-checking, confidence scores |
| LLM10 | Unbounded Consumption | Rate limiting, token budgets, timeout enforcement |

---

## Prompt Injection

### Tipos

- **Direct**: El usuario inyecta instrucciones en su input
- **Indirect**: Datos externos (web, documentos) contienen instrucciones maliciosas

### Mitigación

```python
# 1. Separación estricta system/user
messages = [
    {"role": "system", "content": system_prompt},  # No user-controlled
    {"role": "user", "content": sanitize(user_input)},
]

# 2. Input sanitization
def sanitize(text: str) -> str:
    patterns = [
        r"ignore previous instructions",
        r"forget your instructions",
        r"you are now",
        r"system prompt:",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)
    return text

# 3. Guardrails
from guardrails import Guard
guard = Guard.from_pydantic(output_type=SafeResponse)
result = guard(llm.generate, prompt=user_input)
```

---

## Data Poisoning

- Validar datos de entrenamiento y fine-tuning
- Monitorear drift en embeddings
- Auditar fuentes de datos para RAG
- Checksums en datasets

---

## Supply Chain Security

```bash
# Escanear dependencias
uv pip audit

# Generar SBOM
pip-licenses --format=json > sbom.json

# Pin exact versions en uv.lock
uv lock
```

- Usar solo packages de fuentes confiables
- Verificar signatures de modelos descargados
- No ejecutar modelos no verificados
- Auditar MCP servers de terceros

---

## Secrets Management

```python
# NUNCA esto:
API_KEY = "sk-abc123"  # PROHIBIDO

# Siempre esto:
import os
API_KEY = os.environ["OPENAI_API_KEY"]

# O mejor, usar un secret manager:
from infrastructure.secrets import SecretManager
API_KEY = await SecretManager.get("openai-api-key")
```

**Reglas:**
- Secrets en environment variables o secret managers (AWS Secrets Manager, Vault)
- `.env` en `.gitignore` siempre
- Rotar credenciales periódicamente
- Audit trail de acceso a secrets
- Nunca loggear secrets

---

## Guardrails

### NeMo Guardrails (NVIDIA)

Framework para definir conversational rails:

```yaml
# config/rails.yaml
define user ask_about_competitors
  "What do you think about competitor X?"

define bot refuse_competitor_discussion
  "I can only discuss topics related to our products."

define flow
  user ask_about_competitors
  bot refuse_competitor_discussion
```

### Guardrails AI

Validación de outputs con validators:

```python
from guardrails import Guard
from guardrails.hub import ToxicLanguage, PIIFilter

guard = Guard().use_many(
    ToxicLanguage(on_fail="fix"),
    PIIFilter(on_fail="fix"),
)

result = guard(
    llm.generate,
    prompt="Summarize the document",
)
```

---

## API Security

```python
# Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest): ...

# Input validation
from pydantic import BaseModel, constr

class ChatRequest(BaseModel):
    message: constr(max_length=4096)
    model: str = "gpt-4o"

# Authentication
from fastapi.security import HTTPBearer
security = HTTPBearer()
```

---

## Checklist de Seguridad

- [ ] Prompt injection mitigations activas
- [ ] System prompts no expuestos
- [ ] Inputs validados con Pydantic
- [ ] Outputs validados antes de retornar al usuario
- [ ] Secrets en environment variables o secret manager
- [ ] `.env` en `.gitignore`
- [ ] Rate limiting en endpoints públicos
- [ ] PII filtering en logs y outputs
- [ ] Dependencias escaneadas (audit)
- [ ] MCP tools con allowlisting y timeouts
- [ ] Token budgets configurados
- [ ] Human-in-the-loop para acciones críticas

Ver también: [MCP.md](MCP.md), [GOVERNANCE.md](GOVERNANCE.md)
