# AI Context Kit

## Purpose
This directory contains the **Knowledge Base** for your AI Coding Agents (Claude CLI, Gemini CLI, Cursor, Windsurf, etc.).

It is designed to be **portable**. You can copy this entire `ai_context` folder into any new project to instantly give your AI agents the context, rules, and skills they need to generate high-quality, production-ready code.

## How to Use
1. **Copy** this folder to your new project's root.
2. When starting a chat with your AI, tell it:
   > "Read the `ai_context/RULES.md` file first to understand the architectural standards and capabilities."
3. The AI will follow the rules and can look up specific skills in `ai_context/skills/` or code examples in `ai_context/examples/`.

## Structure
- `RULES.md`: The core operating rules and architectural standards.
- `skills/`: Specialized guides (Security, RAG, API, etc.).
- `examples/`: Reference code implementations to prevent hallucinations.
