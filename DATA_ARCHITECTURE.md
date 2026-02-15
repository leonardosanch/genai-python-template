# Arquitecturas de Datos: BI, Big Data & Analytics

Guía de implementación de arquitecturas modernas de datos. Enfoque en Lakehouses, Data Mesh y Real-time Analytics para sistemas GenAI.

---

## 1. Lakehouse Architecture (Estándar Actual)

Unificación de Data Lake (flexibilidad/costos) + Data Warehouse (calidad/performance). ACID sobre object storage.

### Componentes Core
- **Storage**: S3 / ADLS / GCS (Parquet/Delta/Iceberg)
- **Compute**: Spark / Dremio / Databricks / Snowflake
- **Catalog**: Hive Metastore / AWS Glue / Unity Catalog
- **Formato**: Delta Lake (preferido), Apache Iceberg, Apache Hudi

### Implementación Delta Lake (PySpark)

```python
# src/infrastructure/lakehouse/delta_manager.py
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit

class DeltaLakeManager:
    def __init__(self, warehouse_path: str = "s3://datalake/warehouse"):
        self.builder = SparkSession.builder \
            .appName("LakehouseApp") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
        
        self.spark = configure_spark_with_delta_pip(self.builder).getOrCreate()
        self.warehouse_path = warehouse_path

    def write_bronze(self, df, table_name: str, partition_cols: list[str] = None):
        """
        Bronze: Datos crudos, ingestas sin modificaciones.
        Modo append-only, schema enforcement disabled.
        """
        writer = df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true")
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
            
        writer.save(f"{self.warehouse_path}/bronze/{table_name}")
        
        # Register in metastore
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS bronze.{table_name}
            USING DELTA LOCATION '{self.warehouse_path}/bronze/{table_name}'
        """)

    def write_silver(self, df, table_name: str, merge_key: str = "id"):
        """
        Silver: Datos limpios, normalizados, deduplicados.
        Merge (upsert) para idempotencia.
        """
        delta_path = f"{self.warehouse_path}/silver/{table_name}"
        
        if not self.spark._jsparkSession.catalog().tableExists(f"silver.{table_name}"):
            df.write.format("delta").mode("overwrite").save(delta_path)
            self.spark.sql(f"CREATE TABLE silver.{table_name} USING DELTA LOCATION '{delta_path}'")
        else:
            # Merge para idempotencia
            from delta.tables import DeltaTable
            delta_table = DeltaTable.forPath(self.spark, delta_path)
            delta_table.alias("target").merge(
                df.alias("source"),
                f"target.{merge_key} = source.{merge_key}"
            ).whenMatchedUpdateAll() \
             .whenNotMatchedInsertAll() \
             .execute()

    def write_gold(self, df, table_name: str, partition_cols: list[str] = None):
        """
        Gold: Aggregaciones business-ready, optimizadas para BI.
        Z-ordering en columnas de filtro frecuente.
        """
        writer = df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true")
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
            
        writer.save(f"{self.warehouse_path}/gold/{table_name}")
        
        # Optimización post-write
        self.spark.sql(f"""
            OPTIMIZE gold.{table_name} 
            ZORDER BY (customer_id, event_date)
        """)
```

---

## 2. Unstructured Data Lakehouse (GenAI Focused)

Para sistemas RAG y Multimodales, el Lakehouse debe gestionar datos no estructurados.

### Capas para GenAI
| Capa | Contenido | Formato |
|------|-----------|---------|
| **Raw (Bronze)** | PDFs, Imágenes, Audio originales | Binarios en Object Storage |
| **Processed (Silver)** | Texto extraído, Cleaned HTML, Markdown | Parquet / Delta (cols: `raw_text`, `metadata`) |
| **Semántico (Gold)** | Chunks + Embeddings | Vector DB (pinecone) + Delta (metadata enriquecida) |

### Flujo de Ingesta
1. **Landing**: Sube PDF a `s3://bucket/raw/doc.pdf`.
2. **Trigger**: Evento S3 notifica a pipeline (Celery/Kafka).
3. **Processing**: Extrae texto, limpia, guarda en `silver.documents` table.
4. **Embedding**: Genera vectores, guarda en Vector Store y referencia en `gold.embeddings`.

---

## 3. Data Mesh Architecture

Descentralización de la propiedad de datos por dominios, ideal para sistemas multi-agente grandes.

### Principios
1. **Domain-Oriented Ownership**: El equipo de "Marketing" es dueño de sus datos, no el equipo de "Data Platform".
2. **Data as a Product**: Los datasets son productos con SLA, documentación y calidad garantizada.
3. **Self-Serve Infrastructure**: Plataforma común (Spark, S3, Access Control) provista centralmente.
4. **Federated Governance**: Reglas globales (seguridad, PII) pero ejecución local.

### Implementación en este Proyecto
- **Dominio RAG**: Dueño de `documents`, `chunks`, `embeddings`.
- **Dominio Usuarios**: Dueño de `users`, `preferences`, `api_usage`.
- **Interoperabilidad**: Contratos de datos definidos con Pydantic/Schemas.

---

## 4. Real-time Analytics

Arquitecturas para baja latencia.

### Lambda Architecture (Clásica)
- **Batch Layer**: Spark procesa histórico (preciso, lento).
- **Speed Layer**: Flink/Spark Streaming procesa lo reciente (rápido, aproximado).
- **Serving Layer**: Une ambos resultados.

### Kappa Architecture (Moderna/Simplificada)
- **Todo es Stream**.
- Kafka/Redpanda como log inmutable de verdad.
- Reprocesamiento = Replay del histórico del topic.
- **Herramientas**: Flink, Kafka Streams, Spark Structured Streaming.

### Ejemplo: Streaming RAG Updates
Cuando un documento cambia, actualizar el índice vectorial en tiempo real.

```python
# src/infrastructure/streaming/kafka_consumer.py
from pyspark.sql.functions import from_json, col

def process_stream(spark):
    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "documents_updates") \
        .load()

    # Parse JSON
    parsed_df = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    # Write to Delta Table (Silver)
    query = parsed_df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/checkpoints/docs") \
        .table("silver.documents")

    query.awaitTermination()
```

---

## Comparativa de Arquitecturas

| Arquitectura | Latencia | Complejidad | Caso de Uso Ideal |
|--------------|----------|-------------|-------------------|
| **Lakehouse** | Minutos/Horas | Media | BI, Reporting, ML Training |
| **Data Warehouse**| Minutos/Horas | Baja (SQL) | Analítica corporativa tradicional |
| **Data Mesh** | N/A (Organizacional)| Alta | Empresas grandes, múltiples equipos |
| **Lambda/Kappa** | Milisegundos | Alta | Fraude, IoT, Pricing, Chat en vivo |