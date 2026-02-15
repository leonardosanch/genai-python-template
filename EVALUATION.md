# Evaluación de LLMs

## Por qué evaluar

Los LLMs son no-determinísticos. Sin evaluación sistemática:
- No sabes si un cambio de prompt mejora o empeora el output
- No puedes comparar modelos objetivamente
- No detectas regresiones al cambiar configuración

---

## Frameworks

### DeepEval

Framework de testing para LLMs. Se integra con pytest.

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
)

test_case = LLMTestCase(
    input="What is RAG?",
    actual_output="RAG is Retrieval-Augmented Generation...",
    retrieval_context=["RAG combines retrieval with generation..."],
    expected_output="RAG is a pattern that combines retrieval with LLM generation",
)

metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    HallucinationMetric(threshold=0.5),
]

evaluate(test_cases=[test_case], metrics=metrics)
```

### RAGAS

Framework específico para evaluación de RAG pipelines.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": ["What is RAG?"],
    "answer": ["RAG is Retrieval-Augmented Generation..."],
    "contexts": [["RAG combines retrieval with generation..."]],
    "ground_truth": ["RAG is a pattern that combines retrieval with LLM generation"],
})

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)  # {'faithfulness': 0.85, 'answer_relevancy': 0.90, ...}
```

### LangSmith

Plataforma de LangChain para tracing, evaluación y monitoreo.

- Tracing de toda invocación de LLM
- Datasets de evaluación
- Evaluadores custom
- Comparación de experimentos

---

## Métricas

### Métricas de RAG

| Métrica | Qué mide | Rango |
|---------|----------|-------|
| Faithfulness | ¿El output es fiel al contexto? | 0-1 |
| Answer Relevancy | ¿El output responde la pregunta? | 0-1 |
| Context Precision | ¿Los docs recuperados son relevantes? | 0-1 |
| Context Recall | ¿Se recuperaron todos los docs necesarios? | 0-1 |
| Hallucination Rate | % de información inventada | 0-1 (menor = mejor) |

### Métricas Generales

| Métrica | Qué mide |
|---------|----------|
| Correctness | ¿El output es factualmente correcto? |
| Coherence | ¿El output es coherente y bien estructurado? |
| Toxicity | ¿El output contiene contenido tóxico? |
| Bias | ¿El output muestra sesgos? |
| Latency | Tiempo de respuesta |
| Cost | Costo en tokens/USD |

---

## LLM-as-Judge

Usar un LLM para evaluar outputs de otro LLM.

```python
from deepeval.metrics import GEval

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the actual output is factually correct based on the expected output.",
    evaluation_params=["actual_output", "expected_output"],
    threshold=0.7,
)

correctness_metric.measure(test_case)
print(f"Score: {correctness_metric.score}")
print(f"Reason: {correctness_metric.reason}")
```

**Consideraciones:**
- El LLM evaluador debe ser igual o más capaz que el evaluado
- Definir criterios claros y específicos
- Usar rúbricas para consistencia
- Validar el evaluador contra juicio humano

---

## Benchmarks

### Dataset de Evaluación

```python
# Crear dataset de evaluación
eval_cases = [
    {
        "input": "What are the SOLID principles?",
        "expected_output": "SOLID stands for Single Responsibility...",
        "context": ["SOLID principles were introduced by Robert C. Martin..."],
        "tags": ["software-engineering", "fundamentals"],
    },
    # ... más casos
]

# Guardar como fixture
@pytest.fixture
def eval_dataset():
    return load_eval_cases("tests/evaluation/datasets/software_eng.json")
```

### Ejecutar Benchmarks

```bash
# Ejecutar evaluación completa
uv run pytest tests/evaluation/ -m evaluation -v --tb=long

# Comparar modelos
uv run python scripts/benchmark.py --models gpt-4o,claude-3-5-sonnet --dataset tests/evaluation/datasets/
```

---

## Integración con CI/CD

```yaml
# Evaluation job (semanal, no en cada commit)
evaluation:
  runs-on: ubuntu-latest
  schedule:
    - cron: "0 0 * * 0"  # Domingos
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
    - run: uv sync --frozen
    - run: uv run pytest tests/evaluation/ -m evaluation --tb=long
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    - uses: actions/upload-artifact@v4
      with:
        name: evaluation-results
        path: evaluation_results/
```

---

## Reglas

1. **Evaluar antes de cambiar prompts** — baseline primero
2. **Métricas cuantitativas** — no "se ve bien"
3. **Datasets representativos** — cubrir edge cases
4. **Evaluación periódica** — los modelos cambian con updates
5. **Versionar datasets** — son artefactos tan importantes como el código

Ver también: [TESTING.md](TESTING.md), [RAG.md](RAG.md), [PROMPTS.md](PROMPTS.md)
