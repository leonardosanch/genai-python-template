# RAG (Retrieval-Augmented Generation)

## Qué es RAG

Patrón que mejora las respuestas de LLMs proporcionando contexto relevante recuperado de fuentes externas. Reduce alucinaciones y permite respuestas basadas en datos actualizados.

---

## Patrones RAG

### Naive RAG

Pipeline lineal: query → retrieve → generate.

```python
async def naive_rag(query: str) -> str:
    # 1. Retrieve
    docs = await vector_store.search(query, top_k=5)
    # 2. Generate
    context = "\n".join(doc.content for doc in docs)
    return await llm.generate(f"Context: {context}\n\nQuestion: {query}")
```

**Limitaciones**: Sin reranking, sin query transformation, calidad dependiente del embedding.

### Advanced RAG

Agrega pre-retrieval y post-retrieval processing.

```
Query → Query Transform → Retrieve → Rerank → Filter → Generate → Validate
```

```python
async def advanced_rag(query: str) -> RAGResponse:
    # Pre-retrieval: query transformation
    expanded_queries = await llm.generate_structured(
        prompt=f"Generate 3 alternative queries for: {query}",
        schema=QueryExpansion,
    )

    # Retrieve con múltiples queries
    all_docs = []
    for q in [query] + expanded_queries.alternatives:
        docs = await vector_store.search(q, top_k=5)
        all_docs.extend(docs)

    # Post-retrieval: reranking
    reranked = await reranker.rerank(query, deduplicate(all_docs), top_k=5)

    # Generate con contexto rerankeado
    context = format_context(reranked)
    answer = await llm.generate_structured(
        prompt=build_rag_prompt(query, context),
        schema=RAGResponse,
    )
    return answer
```

### Modular RAG

Componentes intercambiables y composables.

```python
class RAGPipeline:
    def __init__(
        self,
        query_transformer: QueryTransformer,
        retriever: RetrieverPort,
        reranker: RerankerPort | None,
        generator: LLMPort,
        validator: OutputValidator | None,
    ):
        self._query_transformer = query_transformer
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._validator = validator

    async def query(self, input: str) -> RAGResponse:
        queries = await self._query_transformer.transform(input)
        docs = await self._retriever.retrieve(queries)
        if self._reranker:
            docs = await self._reranker.rerank(input, docs)
        response = await self._generator.generate_structured(
            prompt=build_rag_prompt(input, docs),
            schema=RAGResponse,
        )
        if self._validator:
            await self._validator.validate(response, docs)
        return response
```

---

## GraphRAG

Combina knowledge graphs con RAG. En lugar de buscar solo por similitud vectorial, recupera entidades y relaciones de un grafo de conocimiento.

```
Query → Extract Entities → Graph Traversal → Retrieve Subgraph → Enrich with Vectors → Generate
```

```python
from neo4j import AsyncGraphDatabase

class GraphRAGPipeline:
    """RAG que combina knowledge graph con vector search."""

    def __init__(
        self,
        graph_db: AsyncGraphDatabase,
        vector_store: VectorStorePort,
        llm: LLMPort,
    ):
        self._graph = graph_db
        self._vector_store = vector_store
        self._llm = llm

    async def query(self, question: str) -> GraphRAGResponse:
        # 1. Extraer entidades de la pregunta
        entities = await self._llm.generate_structured(
            prompt=f"Extract key entities from: {question}",
            schema=ExtractedEntities,
        )

        # 2. Traversal del grafo — recuperar subgrafo relevante
        subgraph = await self._traverse_graph(entities)

        # 3. Enriquecer con vector search
        vector_docs = await self._vector_store.search(question, top_k=5)

        # 4. Combinar contexto
        context = self._merge_context(subgraph, vector_docs)

        # 5. Generar respuesta con razonamiento
        return await self._llm.generate_structured(
            prompt=self._build_prompt(question, context),
            schema=GraphRAGResponse,
        )

    async def _traverse_graph(self, entities: ExtractedEntities) -> Subgraph:
        """Recuperar nodos y relaciones relevantes del knowledge graph."""
        query = """
        MATCH (n)-[r*1..3]-(related)
        WHERE n.name IN $entity_names
        RETURN n, r, related
        LIMIT 50
        """
        async with self._graph.session() as session:
            result = await session.run(query, entity_names=entities.names)
            return await self._parse_subgraph(result)

    def _merge_context(self, subgraph: Subgraph, docs: list[Document]) -> str:
        """Combinar knowledge graph + vector results en contexto unificado."""
        graph_context = "\n".join(
            f"- {rel.source} --[{rel.type}]--> {rel.target}"
            for rel in subgraph.relationships
        )
        doc_context = "\n".join(doc.content for doc in docs)
        return f"## Knowledge Graph\n{graph_context}\n\n## Documents\n{doc_context}"
```

**Cuándo usar GraphRAG:**
- Datos con relaciones complejas (organigramas, arquitecturas, dependencias)
- Preguntas que requieren traversal multi-hop
- Sistemas donde las entidades y sus relaciones son el core del conocimiento

**Knowledge Graph stores:**
- **Neo4j**: Líder en graph databases, Cypher query language
- **Amazon Neptune**: Managed graph DB en AWS
- **ArangoDB**: Multi-model (graph + document + key-value)

---

## GRaR (Graph-Retrieval-Augmented Reasoning)

El siguiente nivel después de GraphRAG. No solo recupera del grafo — **razona sobre él**.

```
Pregunta → Análisis → Plan de Exploración → Traversal Dinámico → Razonamiento Multi-step → Respuesta
```

### Qué agrega sobre GraphRAG

| Capacidad | GraphRAG | GRaR |
|-----------|----------|------|
| Recuperar del grafo | Si | Si |
| Multi-step reasoning | No | Si |
| Traversal dinámico (el LLM decide qué explorar) | No | Si |
| Responde "por qué" y "qué pasaría si" | Limitado | Si |
| Causalidad e impact analysis | No | Si |
| Agentic (agentes exploran el grafo) | No | Si |

### Implementación con Agentes

```python
from langgraph.graph import StateGraph
from pydantic import BaseModel

class GRaRState(BaseModel):
    """Estado del agente GRaR."""
    question: str
    reasoning_steps: list[str] = []
    explored_nodes: set[str] = set()
    explored_relationships: list[dict] = []
    current_hypothesis: str | None = None
    final_answer: str | None = None
    confidence: float = 0.0

async def analyze_question(state: GRaRState) -> GRaRState:
    """El LLM analiza la pregunta y planifica qué explorar."""
    plan = await llm.generate_structured(
        prompt=f"""Analyze this question and create an exploration plan.
        Question: {state.question}
        Already explored: {state.explored_nodes}

        What entities and relationships should we explore next?
        What hypothesis are we testing?""",
        schema=ExplorationPlan,
    )
    state.reasoning_steps.append(f"Plan: {plan.description}")
    state.current_hypothesis = plan.hypothesis
    return state

async def explore_graph(state: GRaRState) -> GRaRState:
    """Traversal dinámico — el agente decide qué caminos seguir."""
    # El LLM decide qué query Cypher ejecutar
    cypher = await llm.generate(
        prompt=f"""Generate a Cypher query to test this hypothesis:
        Hypothesis: {state.current_hypothesis}
        Already explored: {state.explored_nodes}
        Generate ONLY the Cypher query, nothing else.""",
    )

    async with graph_db.session() as session:
        result = await session.run(cypher)
        new_data = await result.data()

    state.explored_relationships.extend(new_data)
    state.explored_nodes.update(
        node["name"] for record in new_data for node in record.values() if isinstance(node, dict)
    )
    state.reasoning_steps.append(f"Explored: found {len(new_data)} relationships")
    return state

async def reason(state: GRaRState) -> GRaRState:
    """Razonamiento sobre lo explorado — decidir si seguir o concluir."""
    reasoning = await llm.generate_structured(
        prompt=f"""Based on the exploration so far, reason about the question.

        Question: {state.question}
        Hypothesis: {state.current_hypothesis}
        Evidence found: {state.explored_relationships}
        Reasoning so far: {state.reasoning_steps}

        Either:
        1. Conclude with a final answer (if enough evidence)
        2. Formulate a new hypothesis to explore""",
        schema=ReasoningResult,
    )

    state.reasoning_steps.append(f"Reasoning: {reasoning.explanation}")
    if reasoning.is_conclusive:
        state.final_answer = reasoning.answer
        state.confidence = reasoning.confidence
    else:
        state.current_hypothesis = reasoning.next_hypothesis
    return state

def should_continue(state: GRaRState) -> str:
    """Decidir si seguir explorando o concluir."""
    if state.final_answer:
        return "done"
    if len(state.reasoning_steps) > 10:  # Bounded loops
        return "done"
    return "explore"

# Grafo de estado GRaR
graph = StateGraph(GRaRState)
graph.add_node("analyze", analyze_question)
graph.add_node("explore", explore_graph)
graph.add_node("reason", reason)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "explore")
graph.add_edge("explore", "reason")
graph.add_conditional_edges("reason", should_continue, {
    "explore": "explore",  # Seguir explorando
    "done": "__end__",     # Concluir
})

grar_agent = graph.compile()
```

### Ejemplo de uso

```python
# "Si migramos el servicio de pagos, ¿qué módulos y equipos se afectan?"
result = await grar_agent.ainvoke(GRaRState(
    question="Si migramos el servicio de pagos a microservicio, qué módulos y equipos se afectan?"
))

# El agente:
# 1. Identifica "servicio de pagos" como entidad central
# 2. Explora dependencias directas (→ facturación, checkout, reporting)
# 3. Explora dependencias transitivas (→ contabilidad, analytics)
# 4. Identifica equipos owners de cada módulo afectado
# 5. Analiza impacto y genera respuesta con razonamiento paso a paso
```

### Casos de uso ideales

| Caso de uso | Pregunta ejemplo |
|-------------|-----------------|
| Impact analysis | "¿Qué se rompe si cambio esta API?" |
| Root cause analysis | "¿Por qué falla el servicio X?" |
| Decisiones técnicas | "¿Conviene migrar este monolito?" |
| Compliance | "¿Qué datos PII pasan por este flujo?" |
| Dependency mapping | "¿Qué servicios dependen de esta DB?" |
| Multi-agent planning | Agentes que exploran grafos para coordinar tareas |

### GRaR + Multi-Agent

```python
# Agente 1: Explora el grafo de arquitectura
# Agente 2: Evalúa riesgo de cada impacto encontrado
# Agente 3: Genera plan de migración basado en hallazgos

graph = StateGraph(GRaRMultiAgentState)
graph.add_node("explorer", graph_explorer_agent)
graph.add_node("risk_assessor", risk_assessment_agent)
graph.add_node("planner", migration_planner_agent)

graph.set_entry_point("explorer")
graph.add_edge("explorer", "risk_assessor")
graph.add_edge("risk_assessor", "planner")
```

---

## Evolución de Patrones: RAG → GraphRAG → GRaR

```
RAG (Naive)          → Busca documentos similares, genera respuesta
    ↓
Advanced RAG         → Query transform, reranking, validación
    ↓
GraphRAG             → Recupera del knowledge graph + vectores
    ↓
GRaR                 → Razona sobre el grafo, multi-step, agentic
    ↓
Agentic GRaR         → Múltiples agentes exploran y razonan colaborativamente
```

Elegir el nivel según la complejidad de las preguntas:
- **Preguntas factuales simples** → RAG
- **Preguntas que requieren contexto amplio** → Advanced RAG
- **Preguntas sobre relaciones** → GraphRAG
- **Preguntas de impacto, causalidad, "qué pasaría si"** → GRaR

---

## Chunking

Estrategias para dividir documentos en chunks para embedding.

| Estrategia | Caso de uso | Trade-off |
|------------|-------------|-----------|
| Fixed-size | Documentos homogéneos | Simple pero puede cortar contexto |
| Recursive | Texto general | Buen balance, respeta estructura |
| Semantic | Documentos complejos | Mejor calidad, más costoso |
| Document-based | PDFs, HTML | Respeta estructura del documento |

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = splitter.split_documents(documents)
```

---

## Embeddings

| Modelo | Dimensiones | Uso |
|--------|-------------|-----|
| text-embedding-3-small (OpenAI) | 1536 | General, cost-effective |
| text-embedding-3-large (OpenAI) | 3072 | Alta precisión |
| Cohere embed-v3 | 1024 | Multilingüe |
| BGE / GTE (open source) | 768-1024 | Self-hosted, sin costo API |

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed(texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]
```

---

## Vector Stores

| Store | Tipo | Fortaleza |
|-------|------|-----------|
| Pinecone | Managed | Escala, serverless |
| Weaviate | Managed / Self-hosted | Hybrid search, filtros |
| Qdrant | Self-hosted / Cloud | Performance, filtros avanzados |
| ChromaDB | In-process | Desarrollo local, prototipado |
| pgvector | PostgreSQL extension | Si ya usas PostgreSQL |

```python
# Port en domain
class VectorStorePort(ABC):
    @abstractmethod
    async def upsert(self, documents: list[Document]) -> None: ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[Document]: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...
```

---

## Reranking

Reordenar documentos recuperados por relevancia real (no solo similitud vectorial).

```python
# Con Cohere Rerank
from cohere import AsyncClient

class CohereReranker(RerankerPort):
    async def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        response = await self._client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[doc.content for doc in docs],
            top_n=top_k,
        )
        return [docs[r.index] for r in response.results]
```

---

## Evaluación de RAG

Métricas clave (ver [EVALUATION.md](EVALUATION.md) para detalle):

| Métrica | Qué mide |
|---------|----------|
| Faithfulness | ¿La respuesta es fiel al contexto recuperado? |
| Answer Relevancy | ¿La respuesta contesta la pregunta? |
| Context Precision | ¿Los documentos recuperados son relevantes? |
| Context Recall | ¿Se recuperaron todos los documentos necesarios? |

```python
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy],
)
```

---

## Caching Semántico

Cachear respuestas basándose en similitud semántica del query (no exact match).

```python
class SemanticCache:
    def __init__(self, vector_store: VectorStorePort, threshold: float = 0.95):
        self._store = vector_store
        self._threshold = threshold

    async def get(self, query: str) -> str | None:
        results = await self._store.search(query, top_k=1)
        if results and results[0].score >= self._threshold:
            return results[0].metadata["cached_response"]
        return None

    async def set(self, query: str, response: str) -> None:
        doc = Document(
            content=query,
            metadata={"cached_response": response},
        )
        await self._store.upsert([doc])
```

---

## Anti-patrones

- **Retrieve and pray**: No validar la relevancia del contexto recuperado
- **Chunk everything equally**: Usar la misma estrategia de chunking para todo tipo de documento
- **Ignore metadata**: No usar filtros de metadata para refinar retrieval
- **Skip reranking**: Confiar solo en similitud vectorial
- **No evaluation**: No medir faithfulness ni relevancy

Ver también: [EVALUATION.md](EVALUATION.md), [STREAMING.md](STREAMING.md), [AGENTS.md](AGENTS.md), [DATABASES.md](DATABASES.md)
