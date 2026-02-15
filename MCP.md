# Model Context Protocol (MCP)

## Qué es MCP

Protocolo abierto (Anthropic) que estandariza cómo las aplicaciones de AI se conectan con fuentes de datos y herramientas externas.

MCP sigue un modelo **cliente-servidor**:
- **MCP Host**: La aplicación de AI (tu sistema)
- **MCP Client**: Componente que mantiene conexión con un server
- **MCP Server**: Servicio que expone tools, resources y prompts

---

## Conceptos Clave

### Tools

Funciones que el LLM puede invocar. Equivalente a function calling pero estandarizado.

```python
from mcp.server import Server
from mcp.types import Tool

server = Server("my-server")

@server.tool()
async def search_database(query: str, limit: int = 10) -> str:
    """Search the internal database for relevant documents."""
    results = await db.search(query, limit=limit)
    return format_results(results)
```

### Resources

Datos que el servidor expone al cliente (archivos, DB records, configuración).

```python
@server.resource("config://app")
async def get_config() -> str:
    """Application configuration."""
    return json.dumps(config.to_dict())
```

### Prompts

Templates de prompts predefinidos que el servidor ofrece.

```python
@server.prompt()
async def summarize_prompt(document_type: str) -> str:
    """Generate a summarization prompt for the given document type."""
    return f"Summarize the following {document_type}. Focus on key points..."
```

---

## Seguridad

### Principios

1. **Least Privilege**: Cada server tiene acceso mínimo necesario
2. **Input Validation**: Todo input de tool se valida con schemas estrictos
3. **Output Validation**: Todo output se valida antes de pasar al LLM
4. **Timeouts**: Toda operación tiene timeout explícito
5. **Audit Trail**: Toda invocación se logea

### Allowlisting

```python
# Configuración de tools permitidas por agente
AGENT_TOOL_PERMISSIONS = {
    "researcher": ["search_database", "fetch_url"],
    "writer": ["save_document"],
    "admin": ["*"],  # Solo en desarrollo
}
```

### Validación de Inputs

```python
from pydantic import BaseModel, validator

class SearchInput(BaseModel):
    query: str
    limit: int = 10

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v

    @validator("limit")
    def limit_range(cls, v):
        if not 1 <= v <= 100:
            raise ValueError("Limit must be between 1 and 100")
        return v
```

### Sandboxing

- Tools que ejecutan código: sandbox obligatorio (Docker, gVisor)
- Tools que acceden a filesystem: paths restringidos
- Tools que hacen requests HTTP: allowlist de dominios
- Tools que acceden a DB: queries parametrizadas, read-only cuando posible

---

## Integración con el Proyecto

```
src/infrastructure/mcp/
├── __init__.py
├── client.py          # MCP client wrapper
├── server.py          # MCP server base
├── tools/             # Tool implementations
│   ├── search.py
│   ├── database.py
│   └── filesystem.py
├── resources/         # Resource providers
└── security/          # Validation, allowlisting
    ├── validators.py
    └── permissions.py
```

---

## Patrones

### Tool Composition

Combinar tools simples en operaciones complejas:

```python
@server.tool()
async def research_and_summarize(topic: str) -> str:
    """Search for a topic and generate a summary."""
    results = await search_database(topic)
    summary = await llm.generate(f"Summarize: {results}")
    return summary
```

### Error Handling

```python
from mcp.types import ToolError

@server.tool()
async def risky_operation(input: str) -> str:
    try:
        result = await execute(input)
        return result
    except TimeoutError:
        raise ToolError("Operation timed out after 30s")
    except PermissionError:
        raise ToolError("Insufficient permissions for this operation")
```

---

## Checklist de Seguridad MCP

- [ ] Inputs validados con Pydantic schemas
- [ ] Outputs sanitizados antes de retornar
- [ ] Timeouts configurados en toda tool
- [ ] Permissions explícitas por agente
- [ ] Logging de toda invocación (sin datos sensibles)
- [ ] Rate limiting en tools costosas
- [ ] Sandbox para ejecución de código
- [ ] No se exponen system prompts vía tools

Ver también: [AGENTS.md](AGENTS.md), [SECURITY.md](SECURITY.md)
