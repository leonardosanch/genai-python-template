# Machine Learning

## ML en el Contexto GenAI

No todo es LLMs. Los sistemas GenAI en producción combinan ML clásico con LLMs.

```
Feature Engineering → Model Training → Evaluation → Serving → Monitoring
```

---

## Feature Engineering

```python
import polars as pl
from sklearn.preprocessing import StandardScaler

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("text").str.len_chars().alias("text_length"),
        pl.col("text").str.split(" ").list.len().alias("word_count"),
        pl.col("created_at").dt.hour().alias("hour_of_day"),
        pl.col("created_at").dt.weekday().alias("day_of_week"),
        pl.col("category").cast(pl.Categorical).to_physical().alias("category_encoded"),
    )
```

---

## Model Training

### scikit-learn

ML clásico: clasificación, regresión, clustering.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Pipeline reproducible
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_weighted")
print(f"F1 Score: {scores.mean():.3f} ± {scores.std():.3f}")

# Entrenar y evaluar
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))
```

### XGBoost

Gradient boosting para datos tabulares.

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=50,
    eval_metric="logloss",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

### PyTorch

Deep learning, NLP custom, fine-tuning.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).mean(dim=1)
        return self.fc(embedded)

# Training loop
model = TextClassifier(vocab_size=10000, embed_dim=256, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```

### Fine-tuning de LLMs

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# LoRA fine-tuning (eficiente en memoria)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()
```

---

## MLflow — Experiment Tracking & Registry

```python
import mlflow

mlflow.set_experiment("document-classifier")

with mlflow.start_run():
    # Log parámetros
    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": 100,
        "max_depth": 6,
    })

    # Entrenar
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Log métricas
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, average="weighted"),
        "precision": precision_score(y_test, predictions, average="weighted"),
    })

    # Log modelo
    mlflow.sklearn.log_model(model, "model", registered_model_name="doc-classifier")
```

---

## Model Serving

### Con FastAPI

```python
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("models/classifier.joblib")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    features = [request.features]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()
    return PredictionResponse(prediction=int(prediction), probability=float(probability))
```

### Con MLflow Serving

```bash
mlflow models serve -m "models:/doc-classifier/Production" -p 5001
```

---

## ML + LLM: Patrones Híbridos

| Patrón | ML | LLM | Ejemplo |
|--------|-----|-----|---------|
| Classification + Generation | Clasifica intent | Genera respuesta | Chatbot con routing |
| Scoring + Explanation | Calcula score | Explica el score | Credit scoring con explicación |
| Anomaly Detection + Analysis | Detecta anomalía | Analiza causa | Monitoreo de sistemas |
| Recommendation + Personalization | Genera ranking | Personaliza presentación | E-commerce |
| Embedding + RAG | Genera embeddings | Genera respuesta | Búsqueda semántica |

```python
# Ejemplo: Classification + Generation
class HybridPipeline:
    def __init__(self, classifier, llm: LLMPort):
        self._classifier = classifier
        self._llm = llm

    async def process(self, user_input: str) -> Response:
        # ML: clasificar intent
        intent = self._classifier.predict([user_input])[0]

        # LLM: generar respuesta según intent
        prompt = INTENT_PROMPTS[intent].format(input=user_input)
        answer = await self._llm.generate(prompt)

        return Response(intent=intent, answer=answer)
```

---

## Sistemas de Recomendación

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentRecommender:
    """Recomendador basado en embeddings."""

    def __init__(self, vector_store: VectorStorePort):
        self._store = vector_store

    async def recommend(self, user_history: list[str], top_k: int = 10) -> list[Document]:
        # Embedding promedio del historial del usuario
        history_embeddings = await embed(user_history)
        user_profile = np.mean(history_embeddings, axis=0)

        # Buscar items similares al perfil
        return await self._store.search_by_vector(user_profile, top_k=top_k)
```

---

## Reglas

1. **ML clásico cuando alcanza** — no todo necesita un LLM
2. **Experiment tracking siempre** — MLflow o similar
3. **Modelos versionados** — en model registry, no en Git
4. **Evaluación offline antes de deploy** — métricas en test set
5. **Monitoring post-deploy** — data drift, model drift
6. **Feature stores** para features reutilizables en producción
7. **LoRA para fine-tuning** — eficiente en memoria vs full fine-tuning

Ver también: [EVALUATION.md](EVALUATION.md), [DATA_ENGINEERING.md](DATA_ENGINEERING.md), [DEPLOYMENT.md](DEPLOYMENT.md)
