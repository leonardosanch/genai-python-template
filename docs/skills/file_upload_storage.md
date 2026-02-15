---
name: File Upload & Storage
description: Patterns for secure file uploads, MIME validation, and storage abstraction (local/S3).
---

# Skill: File Upload & Storage

## Description

This skill covers secure file upload handling, validation, and storage abstraction for Python backend applications. Use this when accepting user-uploaded files (CVs, documents, images), validating file types and sizes, and persisting files to local filesystem or cloud storage (S3, GCS).

## Executive Summary

**Critical file upload rules:**
- NEVER trust `content_type` from the client — validate MIME type using magic bytes (`python-magic`)
- ALWAYS enforce file size limits BEFORE reading the full file into memory — use streaming uploads
- Abstract storage behind a port (`FileStoragePort`) — adapters for local filesystem and S3
- Generate unique filenames (UUID) — never use the original filename for storage (path traversal risk)
- NEVER serve uploaded files directly from the upload directory — use a dedicated endpoint with auth
- Validate file extensions against an explicit allowlist — reject everything else

**Read full skill when:** Implementing file upload endpoints, validating documents (PDF, DOCX), configuring S3 storage, serving files securely, or handling file attachments.

---

## Versiones y Advertencias de Dependencias

| Dependencia | Versión Mínima | Estabilidad |
|-------------|----------------|-------------|
| python-magic | >= 0.4.27 | ✅ Estable (requiere `libmagic` en el sistema) |
| aiobotocore | >= 2.13.0 | ✅ Estable |
| aiofiles | >= 24.1.0 | ✅ Estable |
| Pillow | >= 10.0.0 | ✅ Estable (solo si se procesan imágenes) |

> ⚠️ **python-magic**: Requiere `libmagic1` instalado en el sistema. En Docker: `apt-get install -y libmagic1`. Sin esta librería del sistema, `python-magic` no funcionará.

> ⚠️ **python-magic vs filetype**: `python-magic` usa `libmagic` (C library, más preciso). `filetype` es pure Python (sin dependencia de sistema, menos preciso). Para validación de seguridad, preferir `python-magic`.

---

## Deep Dive

## Core Concepts

1. **Defense in Depth**: Validate file type at multiple levels — extension allowlist, magic bytes, and content inspection. Never rely on a single check.

2. **Streaming Upload**: For large files, never read the entire file into memory. Use FastAPI's `UploadFile` which wraps `SpooledTemporaryFile` (spills to disk after threshold).

3. **Storage Abstraction**: Define `FileStoragePort` in domain layer. Infrastructure adapters implement it for local filesystem (development) and cloud storage (production). Swapping storage backends requires zero application changes.

4. **Filename Security**: Original filenames from users are untrusted input. They can contain path traversal sequences (`../../etc/passwd`), special characters, or extremely long names. Always generate a UUID-based filename and store the original separately as metadata.

5. **Access Control**: Uploaded files are private by default. Serve them through authenticated endpoints or presigned URLs (S3). Never expose the upload directory via static file serving.

---

## External Resources

### 📁 Libraries & Tools

#### File Handling
- **FastAPI File Upload**: [fastapi.tiangolo.com/tutorial/request-files/](https://fastapi.tiangolo.com/tutorial/request-files/)
    - *Best for*: `UploadFile`, `File(...)`, multi-file uploads
- **python-magic**: [github.com/ahupp/python-magic](https://github.com/ahupp/python-magic)
    - *Best for*: MIME type detection via magic bytes (libmagic wrapper)
- **aiofiles**: [github.com/Tinche/aiofiles](https://github.com/Tinche/aiofiles)
    - *Best for*: Async file I/O operations

#### Cloud Storage
- **aiobotocore (S3)**: [aiobotocore.readthedocs.io](https://aiobotocore.readthedocs.io/)
    - *Best for*: Async S3 operations (upload, download, presigned URLs)
- **boto3 S3 Documentation**: [boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
    - *Best for*: S3 API reference (operations, policies, lifecycle)
- **Google Cloud Storage**: [cloud.google.com/storage/docs](https://cloud.google.com/storage/docs)
    - *Best for*: GCS async client patterns
- **MinIO (S3-compatible)**: [min.io/docs](https://min.io/docs/minio/linux/index.html)
    - *Best for*: Self-hosted S3-compatible storage for development and on-prem

---

### 🛡️ Security

- **OWASP File Upload Cheatsheet**: [cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)
    - *Best for*: Comprehensive file upload security checklist
- **ClamAV (Antivirus)**: [clamav.net](https://www.clamav.net/)
    - *Best for*: Open-source antivirus scanning for uploaded files
- **OWASP Path Traversal**: [owasp.org/www-community/attacks/Path_Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
    - *Best for*: Understanding and preventing path traversal attacks

---

## Decision Trees

### Decision Tree 1: Dónde almacenar archivos

```
¿Dónde guardar los archivos subidos?
├── Desarrollo local / MVP
│   └── Filesystem local (`/data/uploads/`)
│       ├── Simple, sin infra adicional
│       ├── Implementar FileStoragePort → LocalFileStorage adapter
│       └── NUNCA servir directamente — usar endpoint con auth
├── Producción en AWS
│   └── Amazon S3
│       ├── Presigned URLs para acceso directo (evita proxy por tu servidor)
│       ├── Lifecycle policies para auto-cleanup
│       └── Server-side encryption (SSE-S3 o SSE-KMS)
├── Producción en GCP
│   └── Google Cloud Storage
│       ├── Signed URLs equivalentes a presigned
│       └── Uniform bucket-level access
├── On-premise / Self-hosted
│   └── MinIO (S3-compatible)
│       ├── Misma API que S3 → mismo adapter
│       └── Ideal para testing y staging
└── Base de datos (BLOB)
    └── ❌ NUNCA — los archivos no van en la DB
        └── Excepto thumbnails < 100KB como bytea en PostgreSQL
```

### Decision Tree 2: Cómo validar archivos

```
¿Cómo validar un archivo subido?
├── Paso 1: Extensión
│   └── Allowlist explícita: [".pdf", ".docx", ".doc", ".jpg", ".png"]
│       └── Rechazar todo lo demás (NO usar denylist)
├── Paso 2: Tamaño
│   └── Antes de leer el archivo completo
│       ├── Content-Length header (puede ser spoofed, pero útil como first check)
│       └── Leer en chunks y abortar si excede el límite
├── Paso 3: Magic bytes (MIME)
│   └── python-magic sobre los primeros bytes del archivo
│       ├── PDF: debe empezar con %PDF
│       ├── DOCX: debe empezar con PK (ZIP format)
│       └── Rechazar si MIME no coincide con extensión
├── Paso 4: Content inspection (opcional, alto riesgo)
│   └── ¿El archivo contiene contenido ejecutable?
│       ├── ClamAV scan para archivos de usuarios no confiables
│       └── Verificar que PDFs no contengan JavaScript embebido
└── Paso 5: Sanitización (opcional)
    └── Re-procesar el archivo (ej: re-save imagen con Pillow para eliminar metadata/EXIF)
```

---

## Instructions for the Agent

1.  **Storage Abstraction**: Define `FileStoragePort` en domain layer con operaciones `save`, `get`, `delete`, `get_url`. Nunca importar `boto3`, `aiofiles`, o `os.path` fuera de infrastructure.

2.  **Validación en capas**: Siempre validar (1) extensión allowlist, (2) tamaño máximo, (3) magic bytes MIME. Los tres checks son obligatorios. La extensión sola NO es suficiente.

3.  **Filenames**: Generar UUID v4 + extensión original como nombre de archivo. Almacenar nombre original como metadata (en DB, no en filesystem). Nunca usar `pathlib.Path(original_name)` para construir rutas — path traversal.

4.  **Size limits**: Configurar `MAX_UPLOAD_SIZE` via `pydantic-settings`. Valor por defecto: 5MB. Validar ANTES de leer todo el archivo. Usar `request.stream()` o `UploadFile.read(chunk_size)` para streaming.

5.  **Serving files**: Nunca `StaticFiles` sobre el directorio de uploads. Crear endpoint `GET /api/v1/files/{file_id}` con autenticación. Para S3, usar presigned URLs con TTL corto (15 min).

6.  **Docker**: Agregar `libmagic1` al Dockerfile (`apt-get install -y libmagic1`). Montar volumen para uploads locales (`-v ./data/uploads:/app/data/uploads`).

7.  **Testing**: Usar `UploadFile` de FastAPI en tests con `io.BytesIO`. Crear fixtures con archivos reales (PDF mínimo, DOCX mínimo) en `tests/fixtures/`.

---

## Code Examples

### Example 1: Port en Domain Layer

```python
# src/domain/ports/file_storage_port.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredFile:
    """Domain value object for a stored file."""

    file_id: str
    original_name: str
    content_type: str
    size_bytes: int
    storage_path: str


class FileStoragePort(ABC):
    """Domain port for file storage — infrastructure implements this."""

    @abstractmethod
    async def save(
        self,
        file_content: bytes,
        original_name: str,
        content_type: str,
    ) -> StoredFile:
        """Save file and return metadata. Raises FileStorageError on failure."""

    @abstractmethod
    async def get(self, file_id: str) -> tuple[bytes, StoredFile]:
        """Retrieve file content and metadata. Raises FileNotFoundError."""

    @abstractmethod
    async def delete(self, file_id: str) -> None:
        """Delete file from storage."""

    @abstractmethod
    async def get_url(self, file_id: str, expires_in: int = 900) -> str:
        """Get a temporary URL to access the file (presigned URL for S3)."""
```

### Example 2: File Validator (Application Layer)

```python
# src/application/services/file_validator.py
import magic
import structlog

from src.domain.exceptions import FileValidationError

logger = structlog.get_logger()

# Allowlist: extension → expected MIME types
ALLOWED_FILE_TYPES: dict[str, list[str]] = {
    ".pdf": ["application/pdf"],
    ".doc": ["application/msword"],
    ".docx": [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",  # DOCX is a ZIP archive
    ],
    ".jpg": ["image/jpeg"],
    ".jpeg": ["image/jpeg"],
    ".png": ["image/png"],
}

DEFAULT_MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class FileValidator:
    """Validates uploaded files: extension, size, and MIME type."""

    def __init__(self, max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES) -> None:
        self._max_size = max_size_bytes
        self._magic = magic.Magic(mime=True)

    def validate(self, content: bytes, filename: str) -> str:
        """Validate file and return detected MIME type.

        Raises FileValidationError if validation fails.
        """
        # 1. Extension check (allowlist)
        ext = self._get_extension(filename)
        if ext not in ALLOWED_FILE_TYPES:
            raise FileValidationError(
                f"Tipo de archivo no permitido: {ext}. "
                f"Permitidos: {', '.join(ALLOWED_FILE_TYPES.keys())}"
            )

        # 2. Size check
        if len(content) > self._max_size:
            max_mb = self._max_size / (1024 * 1024)
            raise FileValidationError(
                f"Archivo excede el tamaño máximo de {max_mb:.0f}MB"
            )

        if len(content) == 0:
            raise FileValidationError("El archivo está vacío")

        # 3. Magic bytes MIME check
        detected_mime = self._magic.from_buffer(content)
        allowed_mimes = ALLOWED_FILE_TYPES[ext]

        if detected_mime not in allowed_mimes:
            logger.warning(
                "file_mime_mismatch",
                filename=filename,
                extension=ext,
                detected_mime=detected_mime,
                expected_mimes=allowed_mimes,
            )
            raise FileValidationError(
                f"El contenido del archivo no coincide con la extensión {ext}"
            )

        return detected_mime

    @staticmethod
    def _get_extension(filename: str) -> str:
        """Extract lowercase extension from filename."""
        if "." not in filename:
            raise FileValidationError("El archivo no tiene extensión")
        return "." + filename.rsplit(".", 1)[-1].lower()
```

### Example 3: Local Filesystem Adapter

```python
# src/infrastructure/storage/local_file_storage.py
import uuid
from pathlib import Path

import aiofiles
import structlog

from src.domain.exceptions import FileStorageError
from src.domain.ports.file_storage_port import FileStoragePort, StoredFile

logger = structlog.get_logger()


class LocalFileStorage(FileStoragePort):
    """Local filesystem storage adapter — for development."""

    def __init__(self, base_path: str = "./data/uploads") -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        file_content: bytes,
        original_name: str,
        content_type: str,
    ) -> StoredFile:
        file_id = str(uuid.uuid4())
        ext = "." + original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
        storage_name = f"{file_id}{ext}"
        file_path = self._base_path / storage_name

        try:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_content)
        except OSError as exc:
            logger.error("file_save_failed", file_id=file_id, error=str(exc))
            raise FileStorageError(f"Failed to save file: {exc}") from exc

        logger.info(
            "file_saved",
            file_id=file_id,
            original_name=original_name,
            size_bytes=len(file_content),
        )

        return StoredFile(
            file_id=file_id,
            original_name=original_name,
            content_type=content_type,
            size_bytes=len(file_content),
            storage_path=str(file_path),
        )

    async def get(self, file_id: str) -> tuple[bytes, StoredFile]:
        matches = list(self._base_path.glob(f"{file_id}.*"))
        if not matches:
            raise FileNotFoundError(f"File not found: {file_id}")

        file_path = matches[0]
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        stored = StoredFile(
            file_id=file_id,
            original_name=file_path.name,
            content_type="application/octet-stream",
            size_bytes=len(content),
            storage_path=str(file_path),
        )
        return content, stored

    async def delete(self, file_id: str) -> None:
        matches = list(self._base_path.glob(f"{file_id}.*"))
        for path in matches:
            path.unlink(missing_ok=True)
        logger.info("file_deleted", file_id=file_id)

    async def get_url(self, file_id: str, expires_in: int = 900) -> str:
        """Local storage returns an API path — not a presigned URL."""
        return f"/api/v1/files/{file_id}"
```

### Example 4: S3 Storage Adapter

```python
# src/infrastructure/storage/s3_file_storage.py
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from aiobotocore.session import get_session

from src.domain.exceptions import FileStorageError
from src.domain.ports.file_storage_port import FileStoragePort, StoredFile

logger = structlog.get_logger()


class S3FileStorage(FileStoragePort):
    """AWS S3 storage adapter — for production."""

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,  # For MinIO compatibility
    ) -> None:
        self._bucket = bucket_name
        self._region = region
        self._endpoint_url = endpoint_url
        self._session = get_session()

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator:
        async with self._session.create_client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
        ) as client:
            yield client

    async def save(
        self,
        file_content: bytes,
        original_name: str,
        content_type: str,
    ) -> StoredFile:
        file_id = str(uuid.uuid4())
        ext = "." + original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
        key = f"uploads/{file_id}{ext}"

        try:
            async with self._get_client() as client:
                await client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=file_content,
                    ContentType=content_type,
                    Metadata={"original-name": original_name},
                    ServerSideEncryption="AES256",
                )
        except Exception as exc:
            logger.error("s3_upload_failed", file_id=file_id, error=str(exc))
            raise FileStorageError(f"S3 upload failed: {exc}") from exc

        logger.info("file_saved_s3", file_id=file_id, key=key, size_bytes=len(file_content))

        return StoredFile(
            file_id=file_id,
            original_name=original_name,
            content_type=content_type,
            size_bytes=len(file_content),
            storage_path=key,
        )

    async def get(self, file_id: str) -> tuple[bytes, StoredFile]:
        async with self._get_client() as client:
            # List objects with prefix to find the file
            response = await client.list_objects_v2(
                Bucket=self._bucket, Prefix=f"uploads/{file_id}", MaxKeys=1
            )
            contents = response.get("Contents", [])
            if not contents:
                raise FileNotFoundError(f"File not found in S3: {file_id}")

            key = contents[0]["Key"]
            obj = await client.get_object(Bucket=self._bucket, Key=key)
            content = await obj["Body"].read()

            stored = StoredFile(
                file_id=file_id,
                original_name=obj.get("Metadata", {}).get("original-name", key),
                content_type=obj["ContentType"],
                size_bytes=len(content),
                storage_path=key,
            )
            return content, stored

    async def delete(self, file_id: str) -> None:
        async with self._get_client() as client:
            response = await client.list_objects_v2(
                Bucket=self._bucket, Prefix=f"uploads/{file_id}", MaxKeys=1
            )
            for obj in response.get("Contents", []):
                await client.delete_object(Bucket=self._bucket, Key=obj["Key"])
        logger.info("file_deleted_s3", file_id=file_id)

    async def get_url(self, file_id: str, expires_in: int = 900) -> str:
        """Generate a presigned URL for temporary file access."""
        async with self._get_client() as client:
            response = await client.list_objects_v2(
                Bucket=self._bucket, Prefix=f"uploads/{file_id}", MaxKeys=1
            )
            contents = response.get("Contents", [])
            if not contents:
                raise FileNotFoundError(f"File not found in S3: {file_id}")

            key = contents[0]["Key"]
            url = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
```

### Example 5: FastAPI Upload Endpoint

```python
# src/interfaces/api/routes/file_routes.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import Response

from src.application.services.file_validator import FileValidator
from src.domain.ports.file_storage_port import FileStoragePort
from src.domain.exceptions import FileValidationError, FileStorageError

router = APIRouter(prefix="/api/v1/files", tags=["files"])

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(..., description="Archivo a subir (PDF, DOCX, max 5MB)"),
    storage: FileStoragePort = Depends(get_file_storage),
    validator: FileValidator = Depends(get_file_validator),
):
    """Upload a file with validation.

    Validates: extension allowlist, file size, MIME type (magic bytes).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo requerido")

    # Read file content (SpooledTemporaryFile handles memory efficiently)
    content = await file.read()

    # Validate (extension + size + magic bytes)
    try:
        detected_mime = validator.validate(content, file.filename)
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Save to storage
    try:
        stored = await storage.save(
            file_content=content,
            original_name=file.filename,
            content_type=detected_mime,
        )
    except FileStorageError as exc:
        raise HTTPException(status_code=500, detail="Error al guardar el archivo")

    return {
        "file_id": stored.file_id,
        "original_name": stored.original_name,
        "content_type": stored.content_type,
        "size_bytes": stored.size_bytes,
    }


@router.get("/{file_id}")
async def download_file(
    file_id: str,
    storage: FileStoragePort = Depends(get_file_storage),
):
    """Download a file by ID. Requires authentication."""
    try:
        content, stored = await storage.get(file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    return Response(
        content=content,
        media_type=stored.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{stored.original_name}"',
        },
    )


@router.get("/{file_id}/url")
async def get_file_url(
    file_id: str,
    storage: FileStoragePort = Depends(get_file_storage),
):
    """Get a temporary URL to download the file (presigned URL for S3)."""
    try:
        url = await storage.get_url(file_id, expires_in=900)  # 15 minutes
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    return {"url": url, "expires_in_seconds": 900}
```

### Example 6: Settings y Dependency Injection

```python
# src/infrastructure/config/storage_settings.py
from enum import Enum

from pydantic_settings import BaseSettings


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class StorageSettings(BaseSettings):
    """Storage configuration from environment variables."""

    STORAGE_BACKEND: StorageBackend = StorageBackend.LOCAL
    STORAGE_LOCAL_PATH: str = "./data/uploads"

    # S3 settings (only needed when STORAGE_BACKEND=s3)
    S3_BUCKET_NAME: str = ""
    S3_REGION: str = "us-east-1"
    S3_ENDPOINT_URL: str | None = None  # For MinIO

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 5

    model_config = {"env_prefix": "", "case_sensitive": True}


# src/interfaces/api/dependencies/storage.py
from functools import lru_cache

from src.infrastructure.config.storage_settings import StorageSettings, StorageBackend
from src.infrastructure.storage.local_file_storage import LocalFileStorage
from src.infrastructure.storage.s3_file_storage import S3FileStorage
from src.domain.ports.file_storage_port import FileStoragePort


@lru_cache
def get_file_storage() -> FileStoragePort:
    """Factory: returns storage adapter based on configuration."""
    settings = StorageSettings()

    if settings.STORAGE_BACKEND == StorageBackend.S3:
        return S3FileStorage(
            bucket_name=settings.S3_BUCKET_NAME,
            region=settings.S3_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL,
        )

    return LocalFileStorage(base_path=settings.STORAGE_LOCAL_PATH)
```

---

## Anti-Patterns to Avoid

### ❌ Trusting Client Content-Type
**Problem**: Client can send `content_type=application/pdf` for a malicious `.exe` file
**Example**:
```python
# BAD: Trusting the client
@app.post("/upload")
async def upload(file: UploadFile):
    if file.content_type == "application/pdf":  # Spoofable!
        await save(file)
```
**Solution**: Validate with magic bytes
```python
# GOOD: Magic bytes validation
detected_mime = magic.from_buffer(content, mime=True)
if detected_mime not in ALLOWED_MIMES:
    raise HTTPException(400, "Tipo de archivo no válido")
```

### ❌ Using Original Filename for Storage
**Problem**: Path traversal attack — `../../etc/passwd` as filename
**Example**:
```python
# BAD: Path traversal risk
path = f"/uploads/{file.filename}"  # filename = "../../etc/passwd"
```
**Solution**: UUID-based filenames
```python
# GOOD: Safe filename generation
file_id = str(uuid.uuid4())
path = f"/uploads/{file_id}.pdf"
```

### ❌ Reading Entire File into Memory for Large Files
**Problem**: Memory exhaustion with many concurrent uploads
**Example**:
```python
# BAD: Reads entire file at once (100MB file = 100MB RAM)
content = await file.read()
```
**Solution**: Chunk-based reading with size limit enforcement
```python
# GOOD: Streaming with size check
chunks = []
total = 0
while chunk := await file.read(8192):
    total += len(chunk)
    if total > MAX_SIZE:
        raise HTTPException(413, "Archivo demasiado grande")
    chunks.append(chunk)
content = b"".join(chunks)
```

### ❌ Serving Uploads via StaticFiles
**Problem**: Anyone with the URL can access any file — no auth, no access control
**Solution**: Dedicated endpoint with authentication, or presigned URLs with short TTL

### ❌ Storing Files in the Database
**Problem**: PostgreSQL BLOBs degrade performance, backup size explodes, no CDN
**Solution**: Object storage (S3, local filesystem) with metadata in DB

---

## File Upload Checklist

### Validation
- [ ] Extension allowlist (not denylist) configured
- [ ] File size limit enforced BEFORE full read
- [ ] Magic bytes MIME validation with `python-magic`
- [ ] MIME type matches expected extension
- [ ] Empty files rejected
- [ ] `libmagic1` installed in Docker image

### Storage
- [ ] `FileStoragePort` defined in domain layer
- [ ] Local adapter for development
- [ ] S3/GCS adapter for production
- [ ] Factory pattern selects adapter from config
- [ ] Upload directory NOT served as static files
- [ ] S3 server-side encryption enabled

### Security
- [ ] Original filenames NOT used for storage (UUID instead)
- [ ] Path traversal prevention (no user input in file paths)
- [ ] Authenticated download endpoint
- [ ] Presigned URLs with short TTL for S3
- [ ] Content-Disposition header on downloads (forces download vs inline)
- [ ] File metadata stored in DB (original name, uploader, timestamp)

### Testing
- [ ] Upload fixtures with real minimal files (PDF, DOCX)
- [ ] Tests for rejected file types (`.exe`, `.sh`, `.py`)
- [ ] Tests for oversized files (413 response)
- [ ] Tests for MIME mismatch (`.pdf` extension with `.exe` content)
- [ ] Mock storage port in unit tests

### Operations
- [ ] Cleanup policy for orphaned files (Celery Beat or S3 lifecycle)
- [ ] Monitoring: upload count, total storage size, error rate
- [ ] Backup strategy for stored files

---

## Additional References

- [OWASP File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)
- [FastAPI File Uploads Documentation](https://fastapi.tiangolo.com/tutorial/request-files/)
- [python-magic GitHub](https://github.com/ahupp/python-magic)
- [aiobotocore Documentation](https://aiobotocore.readthedocs.io/)
- [MinIO Quickstart](https://min.io/docs/minio/linux/index.html)
- [AWS S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
