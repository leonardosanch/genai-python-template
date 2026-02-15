# Skill: GenAI & RAG

## Description
This skill provides best practices and external resources for building production-ready Generative AI systems with Retrieval-Augmented Generation. Use this when implementing RAG pipelines, evaluating LLM outputs, or optimizing retrieval quality.

## Executive Summary

**Critical RAG rules (apply to all GenAI systems):**
- Reranking is MANDATORY in production — top-k by vector similarity alone is insufficient
- Evaluate retrieval quality BEFORE optimizing generation — bad answers = bad retrieval (not bad prompts)
- All RAG responses MUST include source attribution (document_id, chunk_id, relevance_score)
- Use structured output (Pydantic) for ALL LLM responses — never parse free text
- Consult Decision Tree 1 before implementing RAG — validate that RAG is the correct approach vs fine-tuning or few-shot

**Read full skill when:** Designing RAG pipelines, choosing chunking strategies, selecting vector stores, implementing evaluation metrics, or debugging retrieval quality issues.

---

## Versiones y Dimensiones de Embeddings

### Dependencias

| Dependencia | Versión Mínima | Estabilidad |
|-------------|----------------|-------------|
| langchain | >= 0.2.0 | ⚠️ API cambia frecuentemente |
| langchain-core | >= 0.2.0 | ⚠️ Verificar imports |
| langchain-community | >= 0.2.0 | ⚠️ Paquetes movidos frecuentemente |
| langchain-experimental | N/A | ⚠️ **MUY INESTABLE** - verificar existencia de clases |
| ragas | >= 0.1.0 | ✅ Relativamente estable |
| chromadb | >= 0.4.0 | ✅ Estable |
| pinecone-client | >= 3.0.0 | ✅ Estable (breaking changes en v3) |

> ⚠️ **langchain_experimental**: Las clases en este paquete pueden aparecer, desaparecer o renombrarse sin aviso. Siempre verificar que el import funciona antes de usar.

### Dimensiones de Embeddings por Modelo

| Modelo | Provider | Dimensiones | Notas |
|--------|----------|-------------|-------|
| text-embedding-ada-002 | OpenAI | 1536 | Legacy, usar text-embedding-3 |
| text-embedding-3-small | OpenAI | 1536 | Recomendado para cost-effective |
| text-embedding-3-large | OpenAI | 3072 | Mayor precisión |
| embed-english-v3.0 | Cohere | 1024 | Multilingüe disponible |
| all-MiniLM-L6-v2 | HuggingFace | 384 | Rápido, local |
| all-mpnet-base-v2 | HuggingFace | 768 | Mejor calidad, local |

```python
# Verificar dimensiones antes de crear índice
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

def get_embedding_dim(model: str) -> int:
    if model not in EMBEDDING_DIMENSIONS:
        raise ValueError(f"Modelo desconocido: {model}. Verificar dimensiones en docs del provider.")
    return EMBEDDING_DIMENSIONS[model]
```

---

## Deep Dive

## Core Concepts

1.  **Grounded Generation**: RAG combines retrieval with generation to produce factual, verifiable responses.
2.  **Semantic Search**: Vector embeddings enable similarity-based document retrieval.
3.  **Chunking Strategy**: Document splitting impacts retrieval quality and context window usage.
4.  **Evaluation**: Measure faithfulness, relevancy, and context precision systematically.

---

## External Resources

### 📈 Industry Reports & Trends
- **State of AI Report 2024**: [stateof.ai](https://www.stateof.ai/)
    - *Best for*: Comprehensive annual review of AI research, industry, and safety
- **Gartner Top Strategic Technology Trends 2025**: [gartner.com](https://www.gartner.com/en/articles/top-technology-trends-2025)
    - *Best for*: Agentic AI, AI governance platforms, disinformation security

### 📚 Essential Guides & Documentation

#### OpenAI Resources
- **OpenAI Cookbook**: [cookbook.openai.com](https://cookbook.openai.com/)
    - *Best for*: RAG examples, embeddings, function calling, prompt engineering
- **OpenAI Embeddings Guide**: [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
    - *Best for*: Embedding models, similarity search, clustering
- **OpenAI Fine-tuning Guide**: [platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
    - *Best for*: When to fine-tune vs RAG

#### LangChain Documentation
- **LangChain RAG Tutorial**: [python.langchain.com/docs/tutorials/rag](https://python.langchain.com/docs/tutorials/rag)
    - *Best for*: Building RAG pipelines, document loaders, text splitters
- **LangChain Cheatsheet (2024)**: [github.com/JorisdeJong123/LangChain-Cheatsheet](https://github.com/JorisdeJong123/LangChain-Cheatsheet)
    - *Best for*: Quick reference for chains, agents, and memory
- **LangChain Retrievers**: [python.langchain.com/docs/modules/data_connection/retrievers/](https://python.langchain.com/docs/modules/data_connection/retrievers/)
    - *Best for*: Multi-query, contextual compression, ensemble retrievers
- **LangChain Expression Language (LCEL)**: [python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/)
    - *Best for*: Building composable RAG chains

#### LlamaIndex (GPT Index)
- **LlamaIndex Documentation**: [docs.llamaindex.ai](https://docs.llamaindex.ai/)
    - *Best for*: Advanced RAG patterns, query engines, data connectors
- **LlamaIndex RAG Guide**: [docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/)
    - *Best for*: Ingestion, indexing, querying workflows

---

### 🔬 Research Papers & Advanced Concepts

#### RAG Fundamentals
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
    - [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
    - *Best for*: Original RAG paper, foundational concepts
- **Lost in the Middle** (Liu et al., 2023)
    - [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
    - *Best for*: Understanding context window limitations, document ordering

#### Advanced RAG
- **Self-RAG** (Asai et al., 2023)
    - [arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)
    - *Best for*: Self-reflective retrieval, adaptive RAG
- **RAPTOR** (Sarthi et al., 2024)
    - [arxiv.org/abs/2401.18059](https://arxiv.org/abs/2401.18059)
    - *Best for*: Recursive abstractive processing, tree-based retrieval
- **GraphRAG** (Microsoft, 2024)
    - [microsoft.github.io/graphrag/](https://microsoft.github.io/graphrag/)
    - *Best for*: Knowledge graph-based RAG, entity extraction

---

### 🛠️ RAG Frameworks & Tools

#### Frameworks
- **Haystack**: [haystack.deepset.ai](https://haystack.deepset.ai/)
    - *Best for*: Production RAG pipelines, question answering
- **txtai**: [neuml.github.io/txtai/](https://neuml.github.io/txtai/)
    - *Best for*: Semantic search, embeddings database
- **Canopy** (Pinecone): [github.com/pinecone-io/canopy](https://github.com/pinecone-io/canopy)
    - *Best for*: Context-aware RAG with Pinecone

#### Chunking & Text Processing
- **LangChain Text Splitters**: [python.langchain.com/docs/modules/data_connection/document_transformers/](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
    - *Best for*: Recursive, semantic, token-based splitting
- **Unstructured.io**: [unstructured.io](https://unstructured.io/)
    - *Best for*: Parsing PDFs, HTML, images, tables
- **LlamaParse**: [github.com/run-llama/llama_parse](https://github.com/run-llama/llama_parse)
    - *Best for*: Complex document parsing (tables, figures)

---

### 📊 Evaluation & Metrics

#### RAG Evaluation Frameworks
- **RAGAS** (RAG Assessment): [docs.ragas.io](https://docs.ragas.io/)
    - *Best for*: Faithfulness, answer relevancy, context precision/recall
- **DeepEval**: [docs.confident-ai.com](https://docs.confident-ai.com/)
    - *Best for*: LLM evaluation, hallucination detection, bias testing
- **TruLens**: [trulens.org](https://www.trulens.org/)
    - *Best for*: RAG observability, feedback functions, guardrails
- **LangSmith**: [docs.smith.langchain.com](https://docs.smith.langchain.com/)
    - *Best for*: Tracing, debugging, dataset curation

#### Key Metrics
- **Faithfulness**: Answer grounded in retrieved context
- **Answer Relevancy**: Answer addresses the question
- **Context Precision**: Retrieved docs are relevant
- **Context Recall**: All relevant docs retrieved

---

### 🗄️ Vector Databases

#### Production Vector Stores
- **Pinecone**: [pinecone.io](https://www.pinecone.io/)
    - *Best for*: Managed, serverless, high-performance
- **Weaviate**: [weaviate.io](https://weaviate.io/)
    - *Best for*: Hybrid search (vector + keyword), GraphQL API
- **Qdrant**: [qdrant.tech](https://qdrant.tech/)
    - *Best for*: Open-source, filtering, payload indexing
- **Milvus**: [milvus.io](https://milvus.io/)
    - *Best for*: Scalable, distributed, GPU acceleration
- **ChromaDB**: [trychroma.com](https://www.trychroma.com/)
    - *Best for*: Embedded, local development, simplicity
- **pgvector**: [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
    - *Best for*: PostgreSQL extension, existing Postgres infrastructure

#### Comparison & Selection
- **Vector Database Comparison**: [superlinked.com/vector-db-comparison](https://superlinked.com/vector-db-comparison)
    - *Best for*: Feature comparison, benchmarks

---

### 🎯 Embeddings & Reranking

#### Embedding Models
- **OpenAI Embeddings**: `text-embedding-3-small`, `text-embedding-3-large`
    - *Best for*: High quality, API-based
- **Sentence Transformers**: [sbert.net](https://www.sbert.net/)
    - *Best for*: Open-source, self-hosted, multilingual
- **Cohere Embed**: [cohere.com/embed](https://cohere.com/embed)
    - *Best for*: Multilingual, semantic search
- **Voyage AI**: [voyageai.com](https://www.voyageai.com/)
    - *Best for*: Domain-specific embeddings

#### Reranking
- **Cohere Rerank**: [cohere.com/rerank](https://cohere.com/rerank)
    - *Best for*: Improving retrieval precision
- **Cross-Encoder Models**: [sbert.net/examples/applications/cross-encoder/](https://www.sbert.net/examples/applications/cross-encoder/)
    - *Best for*: Pairwise relevance scoring

---

### 💡 Prompt Engineering

#### Guides & Best Practices
- **OpenAI Prompt Engineering Guide**: [platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering)
    - *Best for*: Strategies, tactics, examples
- **Anthropic Prompt Engineering**: [docs.anthropic.com/claude/docs/prompt-engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
    - *Best for*: Claude-specific techniques
- **Prompt Engineering Guide**: [promptingguide.ai](https://www.promptingguide.ai/)
    - *Best for*: Comprehensive techniques (CoT, ReAct, ToT)
- **Learn Prompting**: [learnprompting.org](https://learnprompting.org/)
    - *Best for*: Interactive tutorials

#### Advanced Techniques
- **Chain-of-Thought (CoT)**: Step-by-step reasoning
- **ReAct**: Reasoning + Acting with tools
- **Tree of Thoughts (ToT)**: Exploring multiple reasoning paths
- **Self-Consistency**: Multiple reasoning paths, majority vote

---

### 📖 Books & Courses

#### Books
- **Building LLM Apps** (Chip Huyen)
    - [huyenchip.com/llm-book](https://huyenchip.com/llm-book)
    - *Best for*: Production LLM applications
- **Hands-On Large Language Models** (Jay Alammar, Maarten Grootendorst)
    - *Best for*: Practical LLM development

#### Courses
- **DeepLearning.AI - LangChain for LLM Application Development**
    - [deeplearning.ai/short-courses/langchain-for-llm-application-development/](https://deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- **DeepLearning.AI - Building and Evaluating Advanced RAG**
    - [deeplearning.ai/short-courses/building-evaluating-advanced-rag/](https://deeplearning.ai/short-courses/building-evaluating-advanced-rag/)

---

### Models & Fine-Tuning
- **Hugging Face Course**: [huggingface.co/learn](https://huggingface.co/learn)

## Decision Trees

### Decision Tree 1: RAG vs Fine-Tuning vs In-Context Learning

Use this to decide the right approach BEFORE writing code.

```
¿El conocimiento que necesitas está en documentos externos?
├── SÍ → ¿Los documentos cambian frecuentemente (semanal o más)?
│   ├── SÍ → RAG (indexar y recuperar dinámicamente)
│   └── NO → ¿El dataset es < 100 ejemplos?
│       ├── SÍ → In-Context Learning (few-shot en el prompt)
│       └── NO → ¿Necesitas cambiar el ESTILO/TONO del modelo?
│           ├── SÍ → Fine-Tuning (cambiar comportamiento del modelo)
│           └── NO → RAG (más flexible, sin costo de training)
└── NO → ¿El modelo ya tiene el conocimiento?
    ├── SÍ → Prompt Engineering (zero-shot o few-shot)
    └── NO → ¿Puedes generar/curar un dataset de entrenamiento?
        ├── SÍ → Fine-Tuning
        └── NO → RAG con documentos sintéticos o curados
```

| Approach | Costo inicial | Costo operativo | Latencia | Actualización |
|----------|--------------|----------------|----------|---------------|
| Prompt Engineering | Bajo | Bajo | Baja | Inmediata |
| In-Context Learning | Bajo | Medio (tokens) | Baja | Inmediata |
| RAG | Medio (infra) | Medio (retrieval + LLM) | Media | Minutos-horas |
| Fine-Tuning | Alto (training) | Bajo (menos tokens) | Baja | Días-semanas |
| RAG + Fine-Tuning | Alto | Medio | Media | Mixto |

**Regla general**: Empezar con RAG. Solo fine-tune si RAG no logra la calidad requerida
después de optimizar retrieval, reranking, y prompts.

### Decision Tree 2: Qué patrón RAG usar

```
¿Qué tipo de preguntas responde tu sistema?
├── Factuales simples ("¿Cuál es la política de X?")
│   └── Naive RAG (retrieve → generate)
├── Factuales con contexto amplio ("Resume todo sobre X")
│   └── Advanced RAG (query expansion + reranking)
├── Sobre relaciones ("¿Qué depende de X?")
│   └── GraphRAG (knowledge graph + vector search)
├── Causalidad / impacto ("¿Qué pasa si cambiamos X?")
│   └── GRaR (agente que razona sobre el grafo)
└── Mixtas / impredecibles
    └── Agentic RAG (router que decide el patrón por query)
```

### Decision Tree 3: Estrategia de chunking

```
¿Qué tipo de documento estás procesando?
├── Texto plano homogéneo (artículos, blogs)
│   └── RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
├── Documentos estructurados (manuales, specs)
│   └── Semantic chunking (split por secciones/headers)
├── PDFs con tablas y figuras
│   └── Document-aware parsing (Unstructured.io / LlamaParse)
├── Código fuente
│   └── Language-aware splitter (por funciones/clases)
└── Conversaciones / chat logs
    └── Split por turns, preservar contexto del thread
```

| Tipo documento | Splitter | chunk_size | overlap | Notas |
|---------------|----------|------------|---------|-------|
| Texto general | Recursive | 1000 | 200 | Default seguro |
| Docs técnicos | Markdown/HTML headers | Por sección | 100 | Respetar estructura |
| Legal/contratos | Semantic | 1500 | 300 | Chunks más grandes para contexto |
| FAQs | Por pregunta | Variable | 0 | Cada Q&A es un chunk |
| Código | Language-aware | Por función | 50 | Preservar imports |

### Decision Tree 4: Selección de vector store

```
¿Cuál es tu escenario?
├── Prototipo / desarrollo local
│   └── ChromaDB (in-process, zero config)
├── Ya usas PostgreSQL en producción
│   └── pgvector (sin infra adicional)
├── Necesitas escala serverless (millones de vectores)
│   └── Pinecone (managed, auto-scaling)
├── Necesitas hybrid search (keyword + vector)
│   └── Weaviate (BM25 + vector nativo)
├── Necesitas filtros avanzados + self-hosted
│   └── Qdrant (filtros payload, open-source)
└── Escala masiva + GPU acceleration
    └── Milvus (distributed, GPU-optimized)
```

---

## Instructions for the Agent

1. **Siempre usar el decision tree antes de elegir approach.** Antes de escribir
   código RAG, consultar "Decision Tree 1" para confirmar que RAG es el approach
   correcto. Documentar la decisión.

2. **Evaluar retrieval quality antes de optimizar generation.** Si las respuestas
   son malas, el problema casi siempre es retrieval (chunks malos, embedding incorrecto,
   sin reranking), NO el prompt. Medir context precision/recall primero.

3. **Prefer structured output (Pydantic) para todas las respuestas RAG.** Usar
   `instructor` o function calling. Nunca parsear texto libre del LLM.

4. **Si la pregunta involucra relaciones, usar GraphRAG.** Preguntas tipo
   "¿qué depende de X?", "¿cómo se relacionan X e Y?" requieren knowledge graph,
   no solo vector search.

5. **Bounded loops en Agentic RAG.** Todo agente que decide dinámicamente cuándo
   retrieval debe tener un límite máximo de iteraciones. Default: 10.

6. **Siempre incluir source attribution.** Toda respuesta RAG debe incluir
   document_id, chunk_id, y relevance_score. Ver governance skill para el patrón
   `ExplainableResponse`.

7. **Reranking es obligatorio en producción.** Top-k por similitud vectorial no es
   suficiente. Siempre agregar un reranker (Cohere, cross-encoder) entre retrieval
   y generation.

---

## Code Examples

### Example 1: Query Router (Agentic RAG)

```python
"""Routes queries to the optimal RAG strategy based on query type."""
from enum import Enum
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    FACTUAL = "factual"           # Simple lookup → Naive RAG
    ANALYTICAL = "analytical"     # Needs broad context → Advanced RAG
    RELATIONAL = "relational"     # About relationships → GraphRAG
    CAUSAL = "causal"             # Impact/causality → GRaR
    CONVERSATIONAL = "conversational"  # Follow-up → use chat history


class ClassifiedQuery(BaseModel):
    """LLM classifies the query before routing."""
    query_type: QueryType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    needs_retrieval: bool = True
    suggested_sources: list[str] = []


ROUTER_PROMPT = """Classify this user query into one of these types:
- factual: Simple fact lookup ("What is the refund policy?")
- analytical: Needs broad context or summary ("Summarize all changes in Q4")
- relational: About relationships between entities ("What depends on service X?")
- causal: Impact analysis or "what if" ("What happens if we remove this API?")
- conversational: Follow-up to previous question, needs chat history

Query: {query}
Chat history (last 3 turns): {history}

Also determine if retrieval is needed (some questions can be answered from chat context).
"""


class QueryRouter:
    """Routes queries to the appropriate RAG pipeline."""

    def __init__(self, llm, naive_rag, advanced_rag, graph_rag, grar):
        self._llm = llm
        self._pipelines = {
            QueryType.FACTUAL: naive_rag,
            QueryType.ANALYTICAL: advanced_rag,
            QueryType.RELATIONAL: graph_rag,
            QueryType.CAUSAL: grar,
        }

    async def route(self, query: str, history: list[str] | None = None) -> str:
        """Classify query and route to optimal pipeline."""
        classification = await self._llm.generate_structured(
            prompt=ROUTER_PROMPT.format(query=query, history=history or []),
            schema=ClassifiedQuery,
        )

        if not classification.needs_retrieval:
            # Answer from context/history without retrieval
            return await self._llm.generate(
                prompt=f"Answer based on conversation context: {query}"
            )

        pipeline = self._pipelines.get(classification.query_type)
        if not pipeline:
            pipeline = self._pipelines[QueryType.FACTUAL]  # Fallback

        return await pipeline.query(query)
```

### Example 2: Adaptive Retrieval with Confidence Check

```python
"""RAG that adapts retrieval strategy based on initial result quality."""
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    documents: list  # Retrieved documents
    avg_score: float  # Average relevance score
    strategy_used: str


class AdaptiveRetriever:
    """Escalates retrieval strategy if initial results are poor.

    Strategy escalation:
    1. Vector search (fast, cheap)
    2. Hybrid search (vector + BM25) if vector score < threshold
    3. Query expansion + reranking if hybrid still poor
    4. Return low-confidence flag if all strategies fail
    """

    SCORE_THRESHOLD = 0.7

    def __init__(self, vector_store, bm25_store, reranker, llm):
        self._vector = vector_store
        self._bm25 = bm25_store
        self._reranker = reranker
        self._llm = llm

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        # Level 1: Vector search
        docs = await self._vector.search(query, top_k=top_k)
        avg_score = self._avg_score(docs)

        if avg_score >= self.SCORE_THRESHOLD:
            return RetrievalResult(docs, avg_score, "vector")

        # Level 2: Hybrid (vector + keyword)
        bm25_docs = await self._bm25.search(query, top_k=top_k)
        merged = self._deduplicate(docs + bm25_docs)
        reranked = await self._reranker.rerank(query, merged, top_k=top_k)
        avg_score = self._avg_score(reranked)

        if avg_score >= self.SCORE_THRESHOLD:
            return RetrievalResult(reranked, avg_score, "hybrid+rerank")

        # Level 3: Query expansion
        expanded = await self._expand_query(query)
        all_docs = []
        for q in expanded:
            all_docs.extend(await self._vector.search(q, top_k=top_k))
        all_docs = self._deduplicate(all_docs + merged)
        reranked = await self._reranker.rerank(query, all_docs, top_k=top_k)
        avg_score = self._avg_score(reranked)

        return RetrievalResult(reranked, avg_score, "expanded+hybrid+rerank")

    async def _expand_query(self, query: str) -> list[str]:
        """Generate alternative queries to improve recall."""
        result = await self._llm.generate(
            prompt=f"Generate 3 alternative phrasings for this search query: {query}"
        )
        return [line.strip("- ") for line in result.strip().split("\n") if line.strip()]

    @staticmethod
    def _avg_score(docs: list) -> float:
        if not docs:
            return 0.0
        scores = [getattr(d, "score", 0.0) for d in docs]
        return sum(scores) / len(scores)

    @staticmethod
    def _deduplicate(docs: list) -> list:
        seen = set()
        unique = []
        for doc in docs:
            doc_id = getattr(doc, "id", id(doc))
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        return unique
```

### Example 3: Chunking Strategy Factory

```python
"""Factory for selecting chunking strategy based on document type."""
from enum import Enum
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
    RecursiveCharacterTextSplitter as CodeSplitter,
)


class DocumentType(str, Enum):
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE_PYTHON = "code_python"
    CODE_JS = "code_js"
    FAQ = "faq"
    LEGAL = "legal"


class ChunkingFactory:
    """Creates the appropriate text splitter based on document type."""

    @staticmethod
    def create(doc_type: DocumentType):
        match doc_type:
            case DocumentType.PLAIN_TEXT:
                return RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " "],
                )
            case DocumentType.MARKDOWN:
                return MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "h1"),
                        ("##", "h2"),
                        ("###", "h3"),
                    ]
                )
            case DocumentType.CODE_PYTHON:
                return CodeSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=1500,
                    chunk_overlap=50,
                )
            case DocumentType.CODE_JS:
                return CodeSplitter.from_language(
                    language=Language.JS,
                    chunk_size=1500,
                    chunk_overlap=50,
                )
            case DocumentType.LEGAL:
                return RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", ". "],
                )
            case DocumentType.FAQ:
                # Each Q&A pair is a chunk — split by double newline
                return RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=0,
                    separators=["\n\n"],
                )
            case _:
                return RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200,
                )
```

### Example 4: RAG Evaluation Pipeline

```python
"""Automated RAG evaluation integrated with CI/CD."""
from dataclasses import dataclass
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


@dataclass
class EvalThresholds:
    """Minimum acceptable scores. Fail CI if below."""
    faithfulness: float = 0.85
    answer_relevancy: float = 0.80
    context_precision: float = 0.75
    context_recall: float = 0.75


@dataclass
class EvalResult:
    scores: dict[str, float]
    passed: bool
    failures: list[str]


def evaluate_rag(
    questions: list[str],
    ground_truths: list[str],
    answers: list[str],
    contexts: list[list[str]],
    thresholds: EvalThresholds | None = None,
) -> EvalResult:
    """Run RAGAS evaluation and check against thresholds."""
    thresholds = thresholds or EvalThresholds()

    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts,
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
    }

    failures = []
    if scores["faithfulness"] < thresholds.faithfulness:
        failures.append(f"faithfulness {scores['faithfulness']:.2f} < {thresholds.faithfulness}")
    if scores["answer_relevancy"] < thresholds.answer_relevancy:
        failures.append(f"answer_relevancy {scores['answer_relevancy']:.2f} < {thresholds.answer_relevancy}")
    if scores["context_precision"] < thresholds.context_precision:
        failures.append(f"context_precision {scores['context_precision']:.2f} < {thresholds.context_precision}")
    if scores["context_recall"] < thresholds.context_recall:
        failures.append(f"context_recall {scores['context_recall']:.2f} < {thresholds.context_recall}")

    return EvalResult(scores=scores, passed=len(failures) == 0, failures=failures)


# Usage in CI
# result = evaluate_rag(questions, ground_truths, answers, contexts)
# if not result.passed:
#     print(f"RAG quality gate FAILED: {result.failures}")
#     sys.exit(1)
```

---

## Evaluation Metrics Comparison

| Metric | What It Measures | When to Use | Tool |
|--------|------------------|-------------|------|
| **Faithfulness** | Answer grounded in context | Always | RAGAS |
| **Answer Relevancy** | Answer addresses question | Always | RAGAS |
| **Context Precision** | Relevant chunks retrieved | Optimize retrieval | RAGAS |
| **Context Recall** | All relevant chunks found | Optimize retrieval | RAGAS |
| **Hallucination Rate** | Fabricated information | Critical applications | DeepEval |
| **Toxicity** | Harmful content | User-facing apps | Guardrails AI |
| **Latency** | Response time | Real-time apps | Custom |
| **Cost per Query** | Token usage | Production | LangSmith |

---

## Cost Optimization Strategies

### 1. Caching
- **Semantic caching**: Cache similar queries (95%+ similarity)
- **Exact caching**: Cache identical queries
- **Savings**: 50-80% for repeated queries

### 2. Chunk Optimization
- **Smaller chunks**: Reduce tokens sent to LLM
- **Optimal size**: 500-1000 tokens per chunk
- **Savings**: 20-30% on LLM costs

### 3. Model Selection
- **Use cheaper models for simple queries**: GPT-3.5 vs GPT-4
- **Router pattern**: Route by complexity
- **Savings**: 30-50% on average

### 4. Batch Processing
- **Batch similar queries**: Reduce API calls
- **Async processing**: Parallel requests
- **Savings**: 10-20% on API overhead

---

## Anti-Patterns to Avoid

### :x: Elegir RAG sin consultar el Decision Tree
**Problem**: Implementar RAG cuando fine-tuning o few-shot sería más apropiado.
Un FAQ estático de 20 preguntas no necesita vector store — few-shot en el prompt es
suficiente y más barato.

**Solution**: Siempre consultar "Decision Tree 1: RAG vs Fine-Tuning vs ICL" antes
de escribir código.

### :x: No Chunking Strategy
**Problem**: Documentos enteros enviados al LLM.
```python
# BAD: No chunking
docs = [Document(page_content=entire_file)]
```
**Solution**: Usar `ChunkingFactory` para seleccionar estrategia por tipo de documento.
```python
# GOOD: Strategy per document type
splitter = ChunkingFactory.create(DocumentType.MARKDOWN)
chunks = splitter.split_documents(docs)
```

### :x: Solo Vector Search sin Reranking
**Problem**: Top-k por similitud vectorial no garantiza relevancia real.
Un chunk con alta similitud coseno puede ser irrelevante para la pregunta.

**Solution**: Siempre agregar reranker en producción. Vector search para recall amplio
(top_k=20), reranker para precision (top_n=5).

### :x: Evaluación "looks good"
**Problem**: No medir faithfulness, answer relevancy, o hallucination rate.
Decidir calidad por inspección manual.

**Solution**: Pipeline de evaluación automatizado con RAGAS en CI/CD.
Definir thresholds mínimos y fallar el build si no se cumplen.

### :x: Mismo Patrón RAG para Todo
**Problem**: Usar Naive RAG para preguntas de impacto/causalidad que requieren
razonamiento multi-step.

**Solution**: Implementar `QueryRouter` que clasifica la pregunta y rutea al
patrón apropiado (Naive, Advanced, GraphRAG, GRaR).

### :x: RAG sin Source Attribution
**Problem**: Retornar respuestas sin indicar de qué documentos provienen.
Imposible verificar, auditar, o debuggear.

**Solution**: Toda respuesta RAG incluye `sources` con document_id, chunk_id,
relevance_score. Ver governance skill para `ExplainableResponse`.

---

## RAG Implementation Checklist

### Data Preparation
- [ ] Documents cleaned and preprocessed
- [ ] Chunking strategy defined (size, overlap)
- [ ] Metadata extracted (source, date, author)
- [ ] Test set created for evaluation

### Embedding & Indexing
- [ ] Embedding model selected (OpenAI, Sentence Transformers)
- [ ] Vector store configured (Pinecone, Weaviate, Qdrant)
- [ ] Index created and populated
- [ ] Similarity search tested

### Retrieval
- [ ] Retrieval strategy chosen (vector, hybrid, reranking)
- [ ] Top-k parameter tuned
- [ ] Retrieval quality measured (context precision/recall)
- [ ] Fallback strategy for no results

### Generation
- [ ] LLM model selected (GPT-4, Claude, Llama)
- [ ] Prompt template optimized
- [ ] Temperature and max_tokens configured
- [ ] Streaming enabled (if needed)

### Evaluation
- [ ] Faithfulness measured (RAGAS)
- [ ] Answer relevancy measured (RAGAS)
- [ ] Hallucination rate checked (DeepEval)
- [ ] Latency measured (p50, p95, p99)
- [ ] Cost per query calculated

### Production
- [ ] Caching implemented (semantic or exact)
- [ ] Rate limiting configured
- [ ] Monitoring and alerts set up
- [ ] Cost tracking enabled
- [ ] A/B testing framework ready

---

## Additional References

### Advanced RAG Techniques
- **Pinecone Learning Center**: [pinecone.io/learn/](https://www.pinecone.io/learn/)
    - *Best for*: Vector search best practices
- **Weaviate Blog**: [weaviate.io/blog](https://weaviate.io/blog)
    - *Best for*: Hybrid search, multi-modal RAG
- **LlamaIndex Advanced Guides**: [docs.llamaindex.ai/en/stable/examples/](https://docs.llamaindex.ai/en/stable/examples/)
    - *Best for*: Advanced indexing strategies

### Chunking Strategies
- **Unstructured.io**: [unstructured.io](https://unstructured.io/)
    - *Best for*: Document parsing and chunking
- **LangChain Text Splitters**: [python.langchain.com/docs/modules/data_connection/document_transformers/](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
    - *Best for*: Various chunking strategies

### Reranking
- **Cohere Rerank**: [docs.cohere.com/docs/reranking](https://docs.cohere.com/docs/reranking)
    - *Best for*: Production reranking
- **Sentence Transformers Cross-Encoders**: [sbert.net/examples/applications/cross-encoder/](https://www.sbert.net/examples/applications/cross-encoder/README.html)
    - *Best for*: Open-source reranking
