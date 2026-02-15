# Skill: Data & ML Engineering

## Description
This skill covers the engineering side of Data Science: building pipelines, serving models, and maintaining data quality in production. Use this when designing ETL workflows, training ML models, or deploying data systems.

## Core Concepts

1.  **Pipeline-as-Code**: Versioned, testable, and idempotent data workflows.
2.  **Experiment Tracking**: Systematic logging of parameters, metrics, and models.
3.  **Data Quality**: Schema validation and anomaly detection at every stage.
4.  **Feature Engineering**: Efficiently transforming raw data into model-ready features.

---

## External Resources

### 📊 Data Processing & Transformation

#### High-Performance DataFrames
- **Polars Documentation**: [docs.pola.rs](https://docs.pola.rs/)
    - *Best for*: High-performance data manipulation, lazy execution, 10-100x faster than Pandas
- **Pandas Documentation**: [pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
    - *Best for*: Standard data manipulation, wide ecosystem
- **Polars vs Pandas Guide**: [pola-rs.github.io/polars-book/](https://pola-rs.github.io/polars-book/)
    - *Best for*: Migration guide, performance comparisons

#### SQL Transformations
- **dbt (Data Build Tool)**: [docs.getdbt.com](https://docs.getdbt.com/)
    - *Best for*: SQL-based transformations, testing, documentation
- **dbt Best Practices**: [docs.getdbt.com/guides/best-practices](https://docs.getdbt.com/guides/best-practices)
    - *Best for*: Modeling, testing, deployment patterns

#### Big Data Processing
- **PySpark Documentation**: [spark.apache.org/docs/latest/api/python/](https://spark.apache.org/docs/latest/api/python/)
    - *Best for*: Distributed processing at TB+ scale
- **Databricks Best Practices**: [docs.databricks.com/best-practices/](https://docs.databricks.com/best-practices/)
    - *Best for*: Spark optimization, Delta Lake

---

### 🤖 Machine Learning & MLOps

#### ML Frameworks
- **scikit-learn User Guide**: [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
    - *Best for*: Classical ML (classification, regression, clustering)
- **XGBoost Documentation**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
    - *Best for*: Gradient boosting, tabular data
- **PyTorch Documentation**: [pytorch.org/docs/](https://pytorch.org/docs/)
    - *Best for*: Deep learning, NLP, fine-tuning
- **TensorFlow Guide**: [tensorflow.org/guide](https://www.tensorflow.org/guide)
    - *Best for*: Production ML, TF Serving

#### Experiment Tracking & Model Registry
- **MLflow Documentation**: [mlflow.org/docs/](https://mlflow.org/docs/latest/index.html)
    - *Best for*: Tracking experiments, model registry, deployment
- **Weights & Biases**: [docs.wandb.ai](https://docs.wandb.ai/)
    - *Best for*: Experiment tracking, hyperparameter tuning, collaboration
- **Neptune.ai**: [docs.neptune.ai](https://docs.neptune.ai/)
    - *Best for*: Metadata store, model monitoring

#### Model Serving
- **FastAPI for ML**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
    - *Best for*: REST API for model serving
- **BentoML**: [docs.bentoml.com](https://docs.bentoml.com/)
    - *Best for*: Model packaging and deployment
- **Ray Serve**: [docs.ray.io/en/latest/serve/](https://docs.ray.io/en/latest/serve/)
    - *Best for*: Scalable model serving, batching

---

### 🔄 Workflow Orchestration

#### Orchestration Platforms
- **Apache Airflow**: [airflow.apache.org/docs/](https://airflow.apache.org/docs/)
    - *Best for*: Complex DAGs, scheduling, monitoring
- **Prefect**: [docs.prefect.io](https://docs.prefect.io/)
    - *Best for*: Modern Python workflows, dynamic DAGs
- **Dagster**: [docs.dagster.io](https://docs.dagster.io/)
    - *Best for*: Asset-based orchestration, data quality
- **Mage.ai**: [docs.mage.ai](https://docs.mage.ai/)
    - *Best for*: Simple ETL, notebook-based workflows

#### Workflow Best Practices
- **Airflow Best Practices**: [airflow.apache.org/docs/apache-airflow/stable/best-practices.html](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
    - *Best for*: DAG design, testing, deployment

---

### ✅ Data Quality & Validation

#### Validation Frameworks
- **Great Expectations**: [docs.greatexpectations.io](https://docs.greatexpectations.io/)
    - *Best for*: Data validation, profiling, documentation
- **Pydantic**: [docs.pydantic.dev](https://docs.pydantic.dev/)
    - *Best for*: Schema validation, type checking
- **Pandera**: [pandera.readthedocs.io](https://pandera.readthedocs.io/)
    - *Best for*: DataFrame validation (Pandas, Polars)

#### Data Observability
- **Monte Carlo**: [docs.montecarlodata.com](https://docs.montecarlodata.com/)
    - *Best for*: Data observability, anomaly detection
- **Soda**: [docs.soda.io](https://docs.soda.io/)
    - *Best for*: Data quality monitoring

---

### 📈 Feature Engineering & Feature Stores

#### Feature Engineering
- **Feature Engineering for Machine Learning** (Alice Zheng, Amanda Casari)
    - *Best for*: Feature creation techniques
- **featuretools**: [featuretools.alteryx.com](https://featuretools.alteryx.com/)
    - *Best for*: Automated feature engineering

#### Feature Stores
- **Feast**: [docs.feast.dev](https://docs.feast.dev/)
    - *Best for*: Open-source feature store
- **Tecton**: [docs.tecton.ai](https://docs.tecton.ai/)
    - *Best for*: Enterprise feature platform
- **AWS SageMaker Feature Store**: [docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
    - *Best for*: AWS-native feature store

---

### 📚 Books & Courses

#### Books
- **Designing Data-Intensive Applications** (Martin Kleppmann)
    - [dataintensive.net](https://dataintensive.net/)
    - *Best for*: Data systems fundamentals
- **Machine Learning Engineering** (Andriy Burkov)
    - *Best for*: Production ML systems
- **Fundamentals of Data Engineering** (Joe Reis, Matt Housley)
    - *Best for*: Modern data engineering

#### Courses
- **Full Stack Deep Learning**: [fullstackdeeplearning.com](https://fullstackdeeplearning.com/)
    - *Best for*: Production ML systems
- **MLOps Specialization** (DeepLearning.AI)
    - [deeplearning.ai/courses/machine-learning-engineering-for-production-mlops/](https://www.deeplearning.ai/courses/machine-learning-engineering-for-production-mlops/)

---

## Instructions for the Agent

1.  **Data Processing**:
    - Use Polars for large datasets (>1GB) due to 10-100x performance improvement
    - Use Pandas for small datasets and when ecosystem compatibility is needed
    - Prefer lazy evaluation (Polars `.lazy()`) for memory efficiency
    - Reference Polars docs for migration from Pandas

2.  **Idempotency**:
    - Ensure data scripts can be re-run without causing duplication
    - Use `upsert` (insert or update) or delete-then-insert patterns
    - Add idempotency keys to prevent duplicate processing
    - Test pipelines with multiple runs

3.  **Experiment Tracking**:
    - Always use MLflow for tracking experiments (params, metrics, models)
    - Log hyperparameters, metrics, and artifacts
    - Use model registry for versioning
    - Reference MLflow docs for best practices

4.  **Schema Validation**:
    - Use Pydantic for API input/output validation
    - Use Pandera for DataFrame schema validation
    - Use Great Expectations for comprehensive data quality
    - Validate at ingestion point, not downstream

5.  **Workflow Orchestration**:
    - Use Airflow for complex, scheduled workflows
    - Use Prefect for dynamic, Python-native workflows
    - Use Dagster for asset-based orchestration
    - Always implement retry logic with exponential backoff

6.  **Model Serving**:
    - Use FastAPI for simple REST APIs
    - Use BentoML for model packaging and deployment
    - Use Ray Serve for scalable, batched serving
    - Implement health checks and monitoring

7.  **Feature Engineering**:
    - Use feature stores (Feast) for feature reuse
    - Version features alongside models
    - Monitor feature drift
    - Document feature definitions

8.  **Data Quality**:
    - Implement data quality checks at every stage
    - Use Great Expectations for profiling and validation
    - Monitor data quality metrics over time
    - Alert on quality degradation

9.  **Production Considerations**:
    - Implement monitoring and alerting
    - Use CI/CD for pipeline deployment
    - Version data, code, and models
    - Implement cost tracking and optimization

---

## Code Examples

### Example 1: Data Pipeline with Great Expectations

```python
# src/pipelines/validated_pipeline.py
import great_expectations as gx
import polars as pl
from pathlib import Path

class DataPipeline:
    """Data pipeline with quality validation."""
    
    def __init__(self):
        self.context = gx.get_context()
        self.suite_name = "data_quality_suite"
    
    async def process(self, input_path: str, output_path: str):
        """Process data with validation."""
        # Load data with Polars (faster than Pandas)
        df = pl.read_csv(input_path)
        
        # Validate
        self._validate_data(df)
        
        # Transform
        df_transformed = self._transform(df)
        
        # Save
        df_transformed.write_parquet(output_path)
        
        return df_transformed
    
    def _validate_data(self, df: pl.DataFrame):
        """Run Great Expectations validation."""
        # Create expectations
        suite = self.context.add_or_update_expectation_suite(self.suite_name)
        
        # Add expectations
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="id")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeUnique(column="id")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="age", min_value=0, max_value=120
            )
        )
        
        # Validate
        results = self.context.run_checkpoint(
            checkpoint_name="quality_check",
            batch_request=df.to_pandas()  # GX needs pandas
        )
        
        if not results.success:
            raise ValueError(f"Data quality failed: {results}")
    
    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data."""
        return df.with_columns([
            pl.col("name").str.to_uppercase().alias("name_upper"),
            (pl.col("age") / 10).alias("age_decade")
        ])
```

### Example 2: ML Experiment Tracking with MLflow

```python
# src/ml/experiment.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class MLExperiment:
    """ML experiment with tracking."""
    
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
    
    async def train_and_track(self, X, y, params: dict):
        """Train model with MLflow tracking."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance
            importance = model.feature_importances_
            for i, imp in enumerate(importance):
                mlflow.log_metric(f"feature_{i}_importance", imp)
            
            print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            return model
```

### Example 3: Data Versioning with DVC

```python
# src/data/versioning.py
import subprocess
from pathlib import Path

class DataVersionControl:
    """Manage data versions with DVC."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def add_and_commit(self, file_path: str, message: str):
        """Add file to DVC and commit."""
        # Add to DVC
        subprocess.run(["dvc", "add", file_path], check=True)
        
        # Git add .dvc file
        dvc_file = f"{file_path}.dvc"
        subprocess.run(["git", "add", dvc_file, ".gitignore"], check=True)
        
        # Git commit
        subprocess.run(["git", "commit", "-m", message], check=True)
        
        # Push to DVC remote
        subprocess.run(["dvc", "push"], check=True)
        
        print(f"✅ Data versioned: {file_path}")
    
    def checkout_version(self, git_tag: str):
        """Checkout specific data version."""
        # Git checkout tag
        subprocess.run(["git", "checkout", git_tag], check=True)
        
        # DVC checkout
        subprocess.run(["dvc", "checkout"], check=True)
        
        print(f"✅ Checked out data version: {git_tag}")
```

---

## Anti-Patterns to Avoid

### ❌ No Data Versioning
**Problem**: Can't reproduce experiments  
**Example**:
```python
# BAD: Overwriting data files
df.to_csv("data/processed.csv")  # Lost previous version
```
**Solution**: Use DVC
```python
# GOOD: Version with DVC
df.to_csv("data/processed_v2.csv")
subprocess.run(["dvc", "add", "data/processed_v2.csv"])
```

### ❌ No Data Quality Checks
**Problem**: Bad data causes model failures  
**Example**:
```python
# BAD: No validation
df = pd.read_csv("data.csv")
model.fit(df)  # What if df has nulls, duplicates?
```
**Solution**: Great Expectations
```python
# GOOD: Validate first
validate_data(df)  # Raises error if quality issues
model.fit(df)
```

### ❌ Manual Experiment Tracking
**Problem**: Lost hyperparameters, can't compare runs  
**Example**:
```python
# BAD: Manual notes
# Tried n_estimators=100, accuracy=0.85
# Tried n_estimators=200, accuracy=0.87
```
**Solution**: MLflow
```python
# GOOD: Automatic tracking
with mlflow.start_run():
    mlflow.log_params({"n_estimators": 100})
    mlflow.log_metric("accuracy", 0.85)
```

### ❌ No Model Monitoring
**Problem**: Model degrades in production, no alerts  
**Solution**: Monitor drift with Evidently AI

---

## Data Engineering Checklist

### Data Pipeline
- [ ] Data versioning configured (DVC, LakeFS)
- [ ] Data quality checks (Great Expectations)
- [ ] Pipeline orchestration (Airflow, Prefect)
- [ ] Error handling and retries
- [ ] Idempotent operations

### ML Workflow
- [ ] Experiment tracking (MLflow, W&B)
- [ ] Model versioning and registry
- [ ] Feature engineering documented
- [ ] Train/test split strategy defined
- [ ] Cross-validation implemented

### Production
- [ ] Model monitoring (drift detection)
- [ ] A/B testing framework
- [ ] Model serving (FastAPI, BentoML)
- [ ] Rollback strategy
- [ ] Cost tracking

### Data Quality
- [ ] Schema validation
- [ ] Null checks
- [ ] Uniqueness constraints
- [ ] Range checks
- [ ] Automated quality reports

---

## Additional References

### Data Versioning
- **DVC Documentation**: [dvc.org/doc](https://dvc.org/doc)
    - *Best for*: Git-like data versioning
- **LakeFS**: [lakefs.io/docs](https://lakefs.io/docs/)
    - *Best for*: Data lake versioning

### Data Quality
- **Great Expectations**: [docs.greatexpectations.io](https://docs.greatexpectations.io/)
    - *Best for*: Data validation and profiling
- **Pandera**: [pandera.readthedocs.io](https://pandera.readthedocs.io/)
    - *Best for*: DataFrame validation

### Model Monitoring
- **Evidently AI**: [docs.evidentlyai.com](https://docs.evidentlyai.com/)
    - *Best for*: ML monitoring and drift detection
- **Arize AI**: [docs.arize.com](https://docs.arize.com/)
    - *Best for*: ML observability platform

### A/B Testing
- **GrowthBook**: [docs.growthbook.io](https://docs.growthbook.io/)
    - *Best for*: Feature flags and A/B testing
- **Statsig**: [docs.statsig.com](https://docs.statsig.com/)
    - *Best for*: Experimentation platform
