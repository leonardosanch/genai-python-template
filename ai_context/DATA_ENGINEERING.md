# Data Engineering

## Rol en Sistemas GenAI

Los datos son el combustible. Sin pipelines confiables no hay RAG, no hay fine-tuning, no hay evaluación.

---

## Pipelines ETL / ELT

### ETL (Extract → Transform → Load)

Transforma datos antes de cargarlos al destino.

```python
# Pipeline ETL con funciones puras
async def etl_pipeline(source: DataSource, destination: DataSink) -> PipelineResult:
    # Extract
    raw_data = await source.extract()

    # Transform
    cleaned = clean_data(raw_data)
    validated = validate_schema(cleaned, schema=DocumentSchema)
    enriched = enrich_with_embeddings(validated)

    # Load
    await destination.load(enriched)

    return PipelineResult(
        records_processed=len(enriched),
        records_failed=len(raw_data) - len(enriched),
    )
```

### ELT (Extract → Load → Transform)

Carga raw data primero, transforma en el destino (data warehouse).

---

## Procesamiento de Datos

### Pandas

Análisis y transformación de datos tabulares.

```python
import pandas as pd

async def process_documents(file_path: str) -> list[Document]:
    df = pd.read_csv(file_path)

    # Limpieza
    df = df.dropna(subset=["content"])
    df["content"] = df["content"].str.strip()
    df = df[df["content"].str.len() > 50]

    # Transformación
    df["word_count"] = df["content"].str.split().str.len()
    df["created_at"] = pd.to_datetime(df["created_at"])

    return [
        Document(content=row["content"], metadata={"source": row["source"]})
        for _, row in df.iterrows()
    ]
```

### Polars

Alternativa a Pandas. Más rápido, mejor manejo de memoria, API más consistente.

```python
import polars as pl

def process_large_dataset(path: str) -> pl.DataFrame:
    # Lazy evaluation — no carga todo en memoria
    df = (
        pl.scan_csv(path)
        .filter(pl.col("content").str.len_chars() > 50)
        .with_columns(
            pl.col("content").str.strip_chars().alias("content"),
            pl.col("content").str.split(" ").list.len().alias("word_count"),
        )
        .sort("created_at", descending=True)
        .collect()  # Ejecuta al final
    )
    return df
```

**Cuándo usar cada uno:**

| Feature | Pandas | Polars |
|---------|--------|--------|
| Datasets < 1GB | Si | Si |
| Datasets > 1GB | Lento / OOM | Si (lazy) |
| Ecosistema / integraciones | Enorme | Creciendo |
| Performance | Moderado | 10-100x más rápido |
| API consistency | Inconsistente | Consistente |
| Parallelismo | No nativo | Si (multi-thread) |

### PySpark

Para procesamiento distribuido a gran escala.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, udf
from pyspark.sql.types import ArrayType, FloatType

spark = SparkSession.builder.appName("genai-pipeline").getOrCreate()

# Leer datos
df = spark.read.parquet("s3://data-lake/documents/")

# Transformar
df_clean = (
    df.filter(length(col("content")) > 50)
    .withColumn("content", col("content").trim())
)

# UDF para embeddings (batch)
@udf(returnType=ArrayType(FloatType()))
def generate_embedding(text):
    return embedding_model.encode(text).tolist()

df_with_embeddings = df_clean.withColumn("embedding", generate_embedding(col("content")))

# Escribir a vector store o data warehouse
df_with_embeddings.write.parquet("s3://data-lake/embeddings/")
```

---

## Orquestación de Workflows

### Airflow

Estándar de la industria para orquestación de pipelines.

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False)
def genai_data_pipeline():

    @task()
    def extract_documents() -> list[dict]:
        """Extraer documentos nuevos de la fuente."""
        return fetch_new_documents(since=yesterday())

    @task()
    def chunk_and_embed(documents: list[dict]) -> list[dict]:
        """Chunking + embedding de documentos."""
        chunks = []
        for doc in documents:
            doc_chunks = chunk_document(doc, max_length=1000)
            for chunk in doc_chunks:
                chunk["embedding"] = embed(chunk["content"])
                chunks.append(chunk)
        return chunks

    @task()
    def load_to_vector_store(chunks: list[dict]):
        """Cargar chunks con embeddings al vector store."""
        vector_store.upsert(chunks)

    @task()
    def validate_index():
        """Validar que el índice es consistente."""
        stats = vector_store.get_stats()
        assert stats["total_vectors"] > 0

    docs = extract_documents()
    chunks = chunk_and_embed(docs)
    load_to_vector_store(chunks) >> validate_index()

genai_data_pipeline()
```

### Prefect

Alternativa moderna a Airflow. Más pythónico, mejor DX.

```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=60)
async def extract():
    return await fetch_documents()

@task
async def transform(docs: list[dict]) -> list[dict]:
    return [chunk_and_embed(doc) for doc in docs]

@task
async def load(chunks: list[dict]):
    await vector_store.upsert(chunks)

@flow(name="rag-indexing-pipeline")
async def indexing_pipeline():
    docs = await extract()
    chunks = await transform(docs)
    await load(chunks)
```

### Dagster

Orquestación orientada a assets (data assets como ciudadanos de primera clase).

```python
from dagster import asset, Definitions

@asset
def raw_documents() -> pd.DataFrame:
    """Documentos crudos desde la fuente."""
    return pd.read_sql("SELECT * FROM documents WHERE processed = false", engine)

@asset
def chunked_documents(raw_documents: pd.DataFrame) -> pd.DataFrame:
    """Documentos divididos en chunks."""
    chunks = []
    for _, row in raw_documents.iterrows():
        for chunk in chunk_text(row["content"]):
            chunks.append({"content": chunk, "source_id": row["id"]})
    return pd.DataFrame(chunks)

@asset
def embedded_documents(chunked_documents: pd.DataFrame) -> pd.DataFrame:
    """Chunks con embeddings generados."""
    chunked_documents["embedding"] = chunked_documents["content"].apply(embed)
    return chunked_documents

defs = Definitions(assets=[raw_documents, chunked_documents, embedded_documents])
```

---

## Streaming de Datos

### Kafka

Procesamiento de eventos en tiempo real.

```python
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

# Productor
async def publish_document_event(document: dict):
    producer = AIOKafkaProducer(bootstrap_servers="localhost:9092")
    await producer.start()
    try:
        await producer.send_and_wait(
            "documents.new",
            value=json.dumps(document).encode(),
        )
    finally:
        await producer.stop()

# Consumidor — procesa documentos nuevos en tiempo real
async def consume_and_index():
    consumer = AIOKafkaConsumer(
        "documents.new",
        bootstrap_servers="localhost:9092",
        group_id="indexing-service",
    )
    await consumer.start()
    try:
        async for msg in consumer:
            document = json.loads(msg.value)
            chunks = chunk_document(document)
            embeddings = await embed_batch([c["content"] for c in chunks])
            await vector_store.upsert(chunks, embeddings)
    finally:
        await consumer.stop()
```

### Redpanda

Drop-in replacement de Kafka. Más simple de operar, compatible con Kafka API.

---

## Data Validation & Quality

### Con Pydantic

```python
from pydantic import BaseModel, field_validator

class RawDocument(BaseModel):
    id: str
    content: str
    source: str
    created_at: datetime

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Content too short")
        return v.strip()

def validate_batch(records: list[dict]) -> tuple[list[RawDocument], list[dict]]:
    valid, invalid = [], []
    for record in records:
        try:
            valid.append(RawDocument(**record))
        except ValidationError as e:
            invalid.append({"record": record, "errors": e.errors()})
    return valid, invalid
```

### Con Great Expectations

```python
import great_expectations as gx

context = gx.get_context()
validator = context.sources.pandas_default.read_csv("data.csv")
validator.expect_column_values_to_not_be_null("content")
validator.expect_column_value_lengths_to_be_between("content", min_value=10)
validator.expect_column_values_to_be_in_set("source", ["web", "api", "manual"])
results = validator.validate()
```

---

## dbt (Data Build Tool)

Transformaciones SQL versionadas con testing.

```sql
-- models/staging/stg_documents.sql
SELECT
    id,
    TRIM(content) AS content,
    source,
    created_at,
    LENGTH(content) AS content_length
FROM {{ source('raw', 'documents') }}
WHERE content IS NOT NULL
    AND LENGTH(TRIM(content)) > 50
```

```python
# Python hooks en dbt
def model(dbt, session):
    df = dbt.ref("stg_documents")
    df["embedding"] = df["content"].apply(generate_embedding)
    return df
```

---

## Pipelines para GenAI

### Indexing Pipeline (RAG)

```
Fuente → Extract → Clean → Chunk → Embed → Upsert Vector Store
```

### Fine-tuning Pipeline

```
Dataset → Validate → Format (JSONL) → Upload → Train → Evaluate → Deploy
```

### Evaluation Pipeline

```
Test Cases → Run Model → Compute Metrics → Compare Baseline → Report
```

---

## Reglas

1. **Pipelines son código** — versionados, testeados, revisados
2. **Idempotencia** — ejecutar un pipeline dos veces produce el mismo resultado
3. **Validación en cada paso** — nunca asumir datos limpios
4. **Observabilidad** — logs, métricas, alertas en cada pipeline
5. **Polars sobre Pandas** para datasets grandes (>100MB)
6. **PySpark** solo cuando Polars no escala (TB+ distribuido)
7. **Retry con backoff** en operaciones de I/O

Ver también: [RAG.md](RAG.md), [DATABASES.md](DATABASES.md), [OBSERVABILITY.md](OBSERVABILITY.md), [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md)
