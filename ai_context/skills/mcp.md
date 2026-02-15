---
name: Model Context Protocol (MCP)
description: Standardizing how AI agents connect to external tools and data resources.
---

# Model Context Protocol (MCP)

## Overview
MCP is an open standard that abstracts the connection between AI models and external systems. It replaces ad-hoc "tool" integrations with a standardized client-server architecture.

## Core Concepts

### 1. MCP Server
A service that exposes capabilities to AI clients.
- **Tools**: Executable functions (e.g., `search_db`, `calculator`).
- **Resources**: Data sources (e.g., file contents, database rows).
- **Prompts**: Reusable prompt templates.

### 2. MCP Client
The AI application (Host) that connects to servers to discover and utilize their capabilities.

## Implementation Guide

### Defining a Tool (MCP Server)

```python
from mcp.server import Server

server = Server("my-data-service")

@server.tool()
async def query_sales_data(region: str) -> str:
    """Get sales metrics for a specific region."""
    # Logic to query DB
    return json.dumps(results)
```

### Security & Compliance
- **Least Privilege**: Agents only get access to specific tools.
- **Sandboxing**: Execute risky tools in isolated environments (Docker, gVisor).
- **Human-in-the-loop**: Require confirmation for sensitive actions (write/delete).

## Best Practices
1.  **Strict Schemas**: Use Pydantic to validate all tool inputs.
2.  **Timeouts**: Enforce timeouts on all tool executions.
3.  **Stateless Servers**: MCP servers should ideally be stateless components.
4.  **Error Handling**: Return structured errors so the LLM knows how to retry or correct.

## External Resources
- [MCP Official Specification](https://modelcontextprotocol.io/)
- [Anthropic MCP SDK](https://github.com/modelcontextprotocol)
