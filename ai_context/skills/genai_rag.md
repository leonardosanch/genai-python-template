# Skill: GenAI & RAG

## Description
This skill provides best practices and external resources for building production-ready Generative AI systems with Retrieval-Augmented Generation. Use this when implementing RAG pipelines, evaluating LLM outputs, or optimizing retrieval quality.

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

## Instructions for the Agent
1.  **Retrieval Evaluation**: When implementing RAG, always consider metrics like "Faithfulness" and "Answer Relevancy" (refer to RAGAS).
2.  **Structuring Output**: Prefer JSON mode or Function Calling for consistent schema. Use `instructor` or `pydantic`.
3.  **Graph Reasoning**: If the query involves complex relationships (e.g., "impact of X on Y across departments"), recommend GraphRAG patterns.

---

## Code Examples

### Example 1: Advanced RAG with Reranking

```python
# src/rag/advanced_pipeline.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

async def create_advanced_rag():
    """RAG with reranking for better precision."""
    # Base retriever
    vectorstore = Pinecone.from_existing_index("docs", embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    # Reranker
    compressor = CohereRerank(model="rerank-english-v2.0", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # QA chain
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        return_source_documents=True
    )
    
    return qa
```

### Example 2: Semantic Caching

```python
# src/rag/semantic_cache.py
from langchain.cache import InMemoryCache
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

class SemanticCache:
    """Cache LLM responses based on semantic similarity."""
    
    def __init__(self, similarity_threshold=0.95):
        self.cache = {}
        self.embeddings = OpenAIEmbeddings()
        self.threshold = similarity_threshold
    
    async def get(self, query: str):
        """Get cached response if semantically similar query exists."""
        query_embedding = await self.embeddings.aembed_query(query)
        
        for cached_query, (cached_embedding, response) in self.cache.items():
            similarity = np.dot(query_embedding, cached_embedding)
            if similarity >= self.threshold:
                return response
        
        return None
    
    async def set(self, query: str, response: str):
        """Cache response with query embedding."""
        query_embedding = await self.embeddings.aembed_query(query)
        self.cache[query] = (query_embedding, response)
```

### Example 3: Hybrid Search (Keyword + Vector)

```python
# src/rag/hybrid_search.py
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Pinecone

async def create_hybrid_retriever(documents):
    """Combine BM25 (keyword) and vector search."""
    # Keyword search (BM25)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    
    # Vector search
    vectorstore = Pinecone.from_documents(documents, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Ensemble (weighted combination)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]  # 40% keyword, 60% vector
    )
    
    return ensemble_retriever
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

### ❌ No Chunking Strategy
**Problem**: Entire documents sent to LLM  
**Example**:
```python
# BAD: No chunking
docs = [Document(page_content=entire_file)]
```
**Solution**: Proper chunking
```python
# GOOD: RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

### ❌ Ignoring Evaluation
**Problem**: No metrics, "looks good" testing  
**Solution**: Systematic evaluation with RAGAS/DeepEval

### ❌ Single Retrieval Strategy
**Problem**: Only vector search, missing keyword matches  
**Solution**: Hybrid search (BM25 + vector)

### ❌ No Reranking
**Problem**: Top-k chunks may not be most relevant  
**Solution**: Use reranker (Cohere, Sentence Transformers)

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
