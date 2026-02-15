---
name: Model Context Protocol (MCP)
description: Standardizing how AI agents connect to external tools and data resources.
---

# Skill: Model Context Protocol (MCP)

## Description

MCP is an open standard that abstracts the connection between AI models and external systems,
replacing ad-hoc tool integrations with a standardized client-server architecture built on
JSON-RPC 2.0. It defines a protocol for tools, resources, and prompts that any compliant
host can discover and invoke. This skill covers server implementation, client integration,
security controls, and production-readiness patterns.

## Executive Summary

**Critical MCP rules:**
- Validate ALL tool inputs and outputs with Pydantic — never trust raw tool results
- Timeout EVERY tool call (default: 30s via `asyncio.wait_for`) — hanging tools block entire agent loop
- Per-agent tool allowlist enforcement — middleware rejects calls not in explicit allowlist (least privilege)
- Human-in-the-loop for destructive tools (write, delete, execute) — require approval before execution
- Log EVERY tool invocation with structured logs — tool name, input (sanitized), duration, success/failure, correlation ID

**Read full skill when:** Implementing MCP servers, integrating MCP clients, configuring tool allowlists, adding human approval gates, or debugging tool call failures.

---

## Versiones y Seguridad de Herramientas

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| mcp | >= 1.0.0 | SDK estable oficial |
| pydantic | >= 2.0.0 | Validación de schemas |
| httpx | >= 0.25.0 | Transport SSE recomendado |

### Validated Tool Input

```python
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    query: str = Field(..., min_length=3)
    limit: int = Field(10, ge=1, le=50)

# ✅ SIEMPRE validar antes de ejecutar la lógica de la herramienta
def my_tool(args: dict):
    validated = ToolInput(**args)
    # logic...
```

---

## Deep Dive

## Core Concepts

1. **MCP Server** — A service that exposes capabilities to AI clients through a well-defined
   protocol. Servers register tools (executable functions), resources (data sources like files
   or database rows), and prompts (reusable prompt templates). Servers should be stateless
   and enforce strict input validation on every exposed capability.

2. **MCP Client (Host)** — The AI application that connects to one or more MCP servers to
   discover and invoke their capabilities. The client manages transport (stdio, HTTP/SSE),
   handles reconnection, and enforces timeouts. Claude Code, LangChain, and other frameworks
   can act as MCP hosts.

3. **Tool Registration and Discovery** — Tools are registered with typed schemas (JSON Schema
   or Pydantic models) that describe their inputs and outputs. Clients discover available
   tools at connection time and can filter them through allowlists before exposing them to
   the LLM.

4. **Transport Layer** — MCP supports multiple transports: stdio (local processes), HTTP with
   Server-Sent Events (remote servers), and WebSocket. The transport is abstracted from the
   protocol layer, allowing the same server to be reached via different mechanisms.

5. **Security Model** — MCP follows a least-privilege approach where each agent receives
   access only to explicitly allowed tools. Sensitive operations require human-in-the-loop
   confirmation. All tool executions are sandboxed when possible (Docker, gVisor, subprocess
   isolation).

6. **JSON-RPC 2.0 Foundation** — MCP messages follow the JSON-RPC 2.0 specification for
   request/response and notification patterns. This provides a well-understood wire format
   with built-in error codes and structured results.

## External Resources

### :zap: Official Specification & SDKs
- [MCP Official Specification](https://modelcontextprotocol.io/)
  *Best for:* Understanding the protocol, message format, and capabilities.
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
  *Best for:* Building MCP servers and clients in Python.
- [Anthropic MCP SDK](https://github.com/modelcontextprotocol)
  *Best for:* Reference implementations and official tooling.

### :shield: Security & Validation
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
  *Best for:* Understanding the underlying wire protocol and error handling.
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
  *Best for:* Input/output validation for tool arguments and results.
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
  *Best for:* Security risks specific to LLM tool use and injection attacks.

### :book: Guides & Examples
- [Claude Code MCP Configuration](https://docs.anthropic.com/en/docs/claude-code)
  *Best for:* Configuring MCP servers in `.claude/settings.json` for Claude Code.
- [MCP Server Examples](https://github.com/modelcontextprotocol/servers)
  *Best for:* Reference server implementations (filesystem, GitHub, Slack, databases).

## Instructions for the Agent

1. **Treat MCP servers as external systems.** Never assume a tool call is fast, safe, or
   deterministic. Apply the same defensive patterns as calling any third-party API:
   timeouts, retries, circuit breakers.

2. **Validate all inputs and outputs.** Every tool argument must be validated with Pydantic
   before execution. Every tool result must be validated before passing to the LLM or
   domain logic. Never trust raw tool output.

3. **Enforce timeouts on every call.** All MCP tool invocations must have explicit timeouts
   (default: 30 seconds, configurable per tool). Use `asyncio.wait_for` or equivalent.
   A hanging tool call must not block the entire agent loop.

4. **Log all tool invocations.** Every tool call must produce a structured log entry with:
   tool name, input arguments (sanitized), execution duration, success/failure status,
   and correlation ID. This is mandatory for observability and audit.

5. **Maintain an explicit allowlist per agent.** Each agent must declare which tools it
   can access. The MCP client middleware must reject any tool call not in the allowlist.
   Never expose the full tool catalog to any single agent.

6. **Never trust tool output without validation.** Tool results can contain injection
   payloads, unexpected schemas, or malicious content. Always validate structure and
   sanitize content before returning to the LLM context.

7. **Sandbox risky tools.** Tools that perform writes, deletions, shell commands, or
   network requests must run in isolated environments (Docker containers, subprocess
   with restricted permissions, gVisor). Require human-in-the-loop for destructive
   operations.

## Code Examples

### MCP Server with Tool Registration and Pydantic Validation

```python
"""MCP server with typed tools and Pydantic validation."""
import json
from pydantic import BaseModel, Field, field_validator
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("data-service")


class QueryInput(BaseModel):
    """Validated input for the query tool."""
    region: str = Field(..., min_length=2, max_length=50)
    metric: str = Field(..., pattern=r"^(revenue|orders|users)$")
    limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("region")
    @classmethod
    def sanitize_region(cls, v: str) -> str:
        """Prevent injection via region parameter."""
        return v.strip().replace(";", "").replace("--", "")


class QueryOutput(BaseModel):
    """Structured tool output."""
    data: list[dict]
    row_count: int
    truncated: bool


@server.tool()
async def query_sales_data(region: str, metric: str, limit: int = 100) -> str:
    """Get sales metrics for a specific region."""
    validated = QueryInput(region=region, metric=metric, limit=limit)

    # Simulate data retrieval
    results = await _fetch_from_db(validated.region, validated.metric, validated.limit)

    output = QueryOutput(
        data=results,
        row_count=len(results),
        truncated=len(results) >= validated.limit,
    )
    return output.model_dump_json()


async def _fetch_from_db(region: str, metric: str, limit: int) -> list[dict]:
    """Placeholder for actual database query."""
    return [{"region": region, "metric": metric, "value": 42}]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### MCP Client with Timeout and Error Handling

```python
"""MCP client connecting via stdio with timeout enforcement."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0


class ToolCallResult(BaseModel):
    """Validated result from a tool invocation."""
    tool_name: str
    success: bool
    data: Any
    duration_ms: float
    error: str | None = None


class MCPClientWrapper:
    """Wraps MCP client with timeout, logging, and validation."""

    def __init__(self, server_command: str, args: list[str], timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self._server_params = StdioServerParameters(command=server_command, args=args)
        self._timeout = timeout
        self._session: ClientSession | None = None

    @asynccontextmanager
    async def connect(self):
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                yield self
                self._session = None

    async def call_tool(self, tool_name: str, arguments: dict) -> ToolCallResult:
        """Call a tool with timeout and structured logging."""
        if not self._session:
            raise RuntimeError("Not connected. Use async with client.connect().")

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=self._timeout,
            )
            duration_ms = (time.monotonic() - start) * 1000

            logger.info(
                "tool_call_success",
                extra={"tool": tool_name, "duration_ms": duration_ms},
            )
            return ToolCallResult(
                tool_name=tool_name, success=True,
                data=result.content, duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error("tool_call_timeout", extra={"tool": tool_name, "timeout": self._timeout})
            return ToolCallResult(
                tool_name=tool_name, success=False,
                data=None, duration_ms=duration_ms,
                error=f"Timeout after {self._timeout}s",
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error("tool_call_error", extra={"tool": tool_name, "error": str(exc)})
            return ToolCallResult(
                tool_name=tool_name, success=False,
                data=None, duration_ms=duration_ms, error=str(exc),
            )
```

### Tool Allowlist Enforcement Middleware

```python
"""Middleware that enforces per-agent tool allowlists."""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ToolNotAllowedError(Exception):
    """Raised when an agent tries to call a tool not in its allowlist."""


@dataclass
class AgentToolPolicy:
    """Defines which tools an agent is permitted to use."""
    agent_id: str
    allowed_tools: set[str] = field(default_factory=set)
    require_confirmation: set[str] = field(default_factory=set)


class ToolAllowlistMiddleware:
    """Enforces tool access control before MCP calls."""

    def __init__(self, policies: list[AgentToolPolicy]):
        self._policies = {p.agent_id: p for p in policies}

    def check_access(self, agent_id: str, tool_name: str) -> bool:
        """Check if the agent is allowed to call this tool.

        Raises ToolNotAllowedError if denied.
        Returns True if allowed, triggers confirmation flow if needed.
        """
        policy = self._policies.get(agent_id)
        if policy is None:
            logger.warning("no_policy_found", extra={"agent_id": agent_id})
            raise ToolNotAllowedError(f"No policy defined for agent '{agent_id}'")

        if tool_name not in policy.allowed_tools:
            logger.warning(
                "tool_access_denied",
                extra={"agent_id": agent_id, "tool": tool_name},
            )
            raise ToolNotAllowedError(
                f"Agent '{agent_id}' is not allowed to call tool '{tool_name}'"
            )

        if tool_name in policy.require_confirmation:
            logger.info(
                "tool_requires_confirmation",
                extra={"agent_id": agent_id, "tool": tool_name},
            )
            return self._request_human_confirmation(agent_id, tool_name)

        return True

    def _request_human_confirmation(self, agent_id: str, tool_name: str) -> bool:
        """Placeholder for human-in-the-loop confirmation."""
        # In production: send to approval queue, Slack, or UI prompt
        logger.info("human_confirmation_requested", extra={
            "agent_id": agent_id, "tool": tool_name,
        })
        raise ToolNotAllowedError(
            f"Tool '{tool_name}' requires human confirmation for agent '{agent_id}'"
        )


# Usage: define policies per agent
policies = [
    AgentToolPolicy(
        agent_id="research-agent",
        allowed_tools={"web_search", "read_document"},
        require_confirmation=set(),
    ),
    AgentToolPolicy(
        agent_id="admin-agent",
        allowed_tools={"web_search", "read_document", "delete_record", "execute_query"},
        require_confirmation={"delete_record", "execute_query"},
    ),
]
middleware = ToolAllowlistMiddleware(policies)
```

### Human-in-the-Loop Confirmation for Dangerous Tools

```python
"""Human-in-the-loop confirmation gate for destructive MCP tool calls."""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """Tracks a pending human approval."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_by: str | None = None


class HumanApprovalGate:
    """Gate that blocks destructive tool calls until a human approves."""

    def __init__(self, timeout_seconds: float = 300.0):
        self._timeout = timeout_seconds
        self._pending: dict[str, ApprovalRequest] = {}
        self._events: dict[str, asyncio.Event] = {}

    async def request_approval(
        self, agent_id: str, tool_name: str, arguments: dict
    ) -> ApprovalRequest:
        """Create an approval request and wait for human resolution."""
        request = ApprovalRequest(
            agent_id=agent_id, tool_name=tool_name, arguments=arguments,
        )
        event = asyncio.Event()
        self._pending[request.request_id] = request
        self._events[request.request_id] = event

        logger.info("approval_requested", extra={
            "request_id": request.request_id,
            "agent_id": agent_id,
            "tool": tool_name,
        })

        # Notify external systems (Slack, UI, webhook)
        await self._notify_reviewers(request)

        try:
            await asyncio.wait_for(event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            request.status = ApprovalStatus.EXPIRED
            logger.warning("approval_expired", extra={"request_id": request.request_id})
        finally:
            self._events.pop(request.request_id, None)
            self._pending.pop(request.request_id, None)

        return request

    def resolve(self, request_id: str, approved: bool, reviewer: str) -> None:
        """Called by human reviewer to approve or deny."""
        request = self._pending.get(request_id)
        if not request:
            return

        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        request.resolved_by = reviewer

        logger.info("approval_resolved", extra={
            "request_id": request_id,
            "status": request.status.value,
            "reviewer": reviewer,
        })

        event = self._events.get(request_id)
        if event:
            event.set()

    async def _notify_reviewers(self, request: ApprovalRequest) -> None:
        """Send notification to human reviewers. Override for real integrations."""
        logger.info("notify_reviewers", extra={
            "request_id": request.request_id,
            "tool": request.tool_name,
        })
```

## Anti-Patterns to Avoid

### :x: Trusting Tool Output Blindly

**Problem:** Passing raw MCP tool results directly into LLM context or domain logic
without validation. Tool output can contain injection payloads, malformed data, or
unexpected schemas.

**Example:**
```python
# BAD: raw result goes straight to LLM
result = await session.call_tool("search", {"query": user_input})
llm_context += result.content[0].text  # Unvalidated
```

**Solution:** Always validate tool output with a Pydantic model or schema check before
using it. Sanitize text content for injection patterns.

### :x: No Timeout on Tool Calls

**Problem:** Tool calls without timeouts can hang indefinitely, blocking the entire
agent loop and consuming resources.

**Example:**
```python
# BAD: no timeout, hangs forever if server is unresponsive
result = await session.call_tool("slow_query", {"table": "large"})
```

**Solution:** Wrap every tool call with `asyncio.wait_for(call, timeout=N)`. Configure
per-tool timeouts based on expected execution time.

### :x: Exposing All Tools to All Agents

**Problem:** Giving every agent access to the full tool catalog violates least privilege.
A compromised or hallucinating agent could invoke destructive tools.

**Example:**
```python
# BAD: agent can call anything
tools = await session.list_tools()
# All tools exposed to LLM without filtering
```

**Solution:** Implement an allowlist middleware that filters available tools per agent
role. Require human confirmation for destructive operations.

### :x: Hardcoding Server Configuration

**Problem:** Embedding server paths, ports, or credentials directly in code makes
deployment inflexible and leaks secrets.

**Example:**
```python
# BAD: hardcoded path and no config management
params = StdioServerParameters(command="/usr/local/bin/my-server", args=["--db=prod"])
```

**Solution:** Load server configuration from environment variables or settings files.
Use the infrastructure layer for all connection details.

## MCP Integration Checklist

### Server Setup
- [ ] Tools registered with typed Pydantic input/output schemas
- [ ] Input validation on all tool arguments (length, pattern, range)
- [ ] Structured error responses with actionable messages
- [ ] Server is stateless (no session-dependent state)
- [ ] Health check endpoint or readiness signal

### Client Integration
- [ ] Timeout configured on every tool call (default 30s)
- [ ] Reconnection logic for transport failures
- [ ] Tool discovery cached with TTL refresh
- [ ] Results validated with Pydantic before use

### Security Controls
- [ ] Per-agent tool allowlist enforced at middleware level
- [ ] Human-in-the-loop gate for destructive tools (write, delete, execute)
- [ ] Tool arguments sanitized against injection attacks
- [ ] Server credentials loaded from environment or secret manager
- [ ] Sandbox execution for risky tools (Docker, subprocess isolation)

### Observability
- [ ] Structured log entry for every tool invocation
- [ ] Duration metrics collected per tool
- [ ] Error rates tracked and alerted
- [ ] Correlation IDs propagated from agent to tool call

### Production Readiness
- [ ] Server configuration externalized (env vars, settings files)
- [ ] Graceful shutdown handling for long-running tool calls
- [ ] Rate limiting on tool invocations to prevent abuse
- [ ] Integration tests covering tool call success, timeout, and error paths
- [ ] Documentation of all exposed tools with expected behavior

## Additional References

- [MCP Official Specification](https://modelcontextprotocol.io/) — Protocol definition,
  transport options, and capability negotiation.
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python
  implementation for building servers and clients.
- [MCP Server Examples](https://github.com/modelcontextprotocol/servers) — Reference
  implementations for filesystem, GitHub, Slack, and database servers.
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) — Wire protocol
  foundation for MCP messages.
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
  — Security risks relevant to LLM tool use and agent systems.
