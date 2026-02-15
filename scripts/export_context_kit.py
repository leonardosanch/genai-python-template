import re
import shutil
from pathlib import Path

# Configuration
SOURCE_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = SOURCE_ROOT / "ai_context"
SKILLS_SRC = SOURCE_ROOT / "docs" / "skills"
EXAMPLES_SRC = SOURCE_ROOT / "src" / "examples"
RULES_SRC = SOURCE_ROOT / "GEMINI.md"


def setup_output_dir():
    """Create a clean output directory."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    print(f"✅ Created {OUTPUT_DIR}")


def copy_skills():
    """Copy skills documentation."""
    dest = OUTPUT_DIR / "skills"
    shutil.copytree(SKILLS_SRC, dest)
    print(f"✅ Copied skills to {dest}")


def copy_examples():
    """Copy code examples."""
    dest = OUTPUT_DIR / "examples"
    shutil.copytree(EXAMPLES_SRC, dest)
    print(f"✅ Copied examples to {dest}")


def process_rules():
    """Copy and process GEMINI.md to RULES.md, fixing links."""
    content = RULES_SRC.read_text()

    # Logic to replace absolute/project-specific links with portable relative links
    # 1. Replace links to docs/skills/X.md -> skills/X.md
    # Pattern: (file:///.../docs/skills/xyz.md) or (docs/skills/xyz.md) -> (skills/xyz.md)

    # Regex for full file paths or relative paths pointing to docs/skills
    content = re.sub(r"\(file://.*?/docs/skills/", "(skills/", content)
    content = re.sub(r"\(docs/skills/", "(skills/", content)

    # 2. Replace PROMPTS.md, AGENTS.md etc links if we had them
    # (not copying root md files for now except essential)
    # Ideally we should copy the root MD files too if they are referenced.
    # For now, let's copy the referenced root files to a 'docs' folder inside
    # ai_context or keep them flat?
    # Simpler: Copy common root MDs to root of ai_context

    root_docs = [
        "PROMPTS.md",
        "AGENTS.md",
        "MCP.md",
        "RAG.md",
        "SECURITY.md",
        "EVALUATION.md",
        "STREAMING.md",
        "API.md",
        "DATABASES.md",
        "DATA_ENGINEERING.md",
        "MACHINE_LEARNING.md",
        "EVENT_DRIVEN.md",
        "AUTOMATION.md",
        "ANALYTICS.md",
        "TOOLS.md",
        "DEPLOYMENT.md",
        "OBSERVABILITY.md",
        "TESTING.md",
        "GOVERNANCE.md",
        "ARCHITECTURE.md",
    ]

    for doc in root_docs:
        src = SOURCE_ROOT / doc
        if src.exists():
            shutil.copy(src, OUTPUT_DIR / doc)
            # No link update needed if they stay in root relative to RULES.md

    dest = OUTPUT_DIR / "RULES.md"
    dest.write_text(content)
    print(f"✅ Processed {RULES_SRC.name} -> {dest.name} (with fixed links)")


def create_readme():
    readme_content = """# AI Context Kit

## Purpose
This directory contains the **Knowledge Base** for your AI Coding Agents
(Claude CLI, Gemini CLI, Cursor, Windsurf, etc.).

It is designed to be **portable**. You can copy this entire `ai_context`
folder into any new project to instantly give your AI agents the context,
rules, and skills they need to generate high-quality, production-ready code.

## How to Use
1. **Copy** this folder to your new project's root.
2. When starting a chat with your AI, tell it:
   > "Read the `ai_context/RULES.md` file first to understand the
   > architectural standards and capabilities."
3. The AI will follow the rules and can look up specific skills in
   `ai_context/skills/` or code examples in `ai_context/examples/`.

## Structure
- `RULES.md`: The core operating rules and architectural standards.
- `skills/`: Specialized guides (Security, RAG, API, etc.).
- `examples/`: Reference code implementations to prevent hallucinations.
"""
    (OUTPUT_DIR / "README.md").write_text(readme_content)
    print("✅ Created README.md")


def main():
    print("📦 Packaging AI Context Kit...")
    setup_output_dir()
    copy_skills()
    copy_examples()
    process_rules()
    create_readme()
    print("\n✨ Done! You can now copy the 'ai_context' folder to any project.")


if __name__ == "__main__":
    main()
