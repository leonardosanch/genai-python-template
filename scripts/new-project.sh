#!/usr/bin/env bash
# new-project.sh — Create a new project from the GenAI template
#
# Copies the project structure (CLAUDE.md, skills, config, empty src layout)
# WITHOUT the example code, so you start with a clean project.
#
# Usage:
#   ./scripts/new-project.sh <target-directory>
#   ./scripts/new-project.sh ~/proyectos/sistema_pos
#   ./scripts/new-project.sh ~/proyectos/sistema_pos --with-examples

set -euo pipefail

# Resolve the template root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Validate arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <target-directory> [--with-examples]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 ~/proyectos/sistema_pos"
    echo "  $0 ~/proyectos/mi-api --with-examples"
    exit 1
fi

TARGET_DIR="$1"
WITH_EXAMPLES="${2:-}"
PROJECT_NAME="$(basename "$TARGET_DIR")"

# Check target doesn't exist
if [ -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error: Directory already exists: $TARGET_DIR${NC}"
    exit 1
fi

echo -e "${CYAN}Creating new project: $PROJECT_NAME${NC}"
echo "  From template: $TEMPLATE_DIR"
echo "  Target: $TARGET_DIR"
echo ""

# Create directory structure
mkdir -p "$TARGET_DIR"

# --- Copy configuration files ---
echo -e "${GREEN}[1/6]${NC} Copying configuration..."
cp "$TEMPLATE_DIR/CLAUDE.md" "$TARGET_DIR/"
cp "$TEMPLATE_DIR/pyproject.toml" "$TARGET_DIR/"
cp "$TEMPLATE_DIR/.env.example" "$TARGET_DIR/"
cp "$TEMPLATE_DIR/.python-version" "$TARGET_DIR/" 2>/dev/null || true
cp "$TEMPLATE_DIR/.gitignore" "$TARGET_DIR/" 2>/dev/null || true

# --- Copy skills documentation ---
echo -e "${GREEN}[2/6]${NC} Copying skills (15 files)..."
mkdir -p "$TARGET_DIR/docs/skills"
cp -r "$TEMPLATE_DIR/docs/skills/"*.md "$TARGET_DIR/docs/skills/"

# Copy examples README if it exists
if [ -d "$TEMPLATE_DIR/docs/skills/examples" ]; then
    cp -r "$TEMPLATE_DIR/docs/skills/examples" "$TARGET_DIR/docs/skills/"
fi

# --- Copy Claude Code config ---
echo -e "${GREEN}[3/6]${NC} Copying Claude Code config (.claude/)..."
if [ -d "$TEMPLATE_DIR/.claude" ]; then
    cp -r "$TEMPLATE_DIR/.claude" "$TARGET_DIR/"
fi

# --- Copy CI/CD ---
echo -e "${GREEN}[4/6]${NC} Copying CI/CD config..."
if [ -d "$TEMPLATE_DIR/.github" ]; then
    cp -r "$TEMPLATE_DIR/.github" "$TARGET_DIR/"
fi
cp "$TEMPLATE_DIR/.pre-commit-config.yaml" "$TARGET_DIR/" 2>/dev/null || true
cp "$TEMPLATE_DIR/sonar-project.properties" "$TARGET_DIR/" 2>/dev/null || true

# --- Copy deploy scaffold ---
if [ -d "$TEMPLATE_DIR/deploy" ]; then
    echo -e "${GREEN}[5/6]${NC} Copying deploy scaffold..."
    cp -r "$TEMPLATE_DIR/deploy" "$TARGET_DIR/"
fi

# --- Create clean src structure ---
echo -e "${GREEN}[6/6]${NC} Creating clean project structure..."
mkdir -p "$TARGET_DIR/src/domain/entities"
mkdir -p "$TARGET_DIR/src/domain/ports"
mkdir -p "$TARGET_DIR/src/domain/value_objects"
mkdir -p "$TARGET_DIR/src/application/use_cases"
mkdir -p "$TARGET_DIR/src/application/dtos"
mkdir -p "$TARGET_DIR/src/application/pipelines"
mkdir -p "$TARGET_DIR/src/infrastructure/config"
mkdir -p "$TARGET_DIR/src/infrastructure/llm"
mkdir -p "$TARGET_DIR/src/infrastructure/database"
mkdir -p "$TARGET_DIR/src/infrastructure/cache"
mkdir -p "$TARGET_DIR/src/infrastructure/storage"
mkdir -p "$TARGET_DIR/src/infrastructure/events"
mkdir -p "$TARGET_DIR/src/infrastructure/observability"
mkdir -p "$TARGET_DIR/src/interfaces/api/routes"
mkdir -p "$TARGET_DIR/src/interfaces/api/middleware"
mkdir -p "$TARGET_DIR/src/interfaces/cli"
mkdir -p "$TARGET_DIR/tests/unit"
mkdir -p "$TARGET_DIR/tests/integration"

# Create __init__.py files
find "$TARGET_DIR/src" -type d -exec touch {}/__init__.py \;
find "$TARGET_DIR/tests" -type d -exec touch {}/__init__.py \;

# Optionally copy example code
if [ "$WITH_EXAMPLES" = "--with-examples" ]; then
    echo -e "${YELLOW}Copying example code from template...${NC}"
    cp -r "$TEMPLATE_DIR/src/" "$TARGET_DIR/src/"
    cp -r "$TEMPLATE_DIR/tests/" "$TARGET_DIR/tests/"
fi

# Update pyproject.toml with new project name
sed -i "s/name = \"genai-python-template\"/name = \"$PROJECT_NAME\"/" "$TARGET_DIR/pyproject.toml"

# Initialize git
cd "$TARGET_DIR"
git init -q
echo -e ""
echo -e "${GREEN}Project created successfully: $TARGET_DIR${NC}"
echo ""
echo "  Structure:"
echo "    CLAUDE.md              — Project rules + skill references (relative paths)"
echo "    docs/skills/           — 15 specialized skills"
echo "    src/domain/            — Pure business logic"
echo "    src/application/       — Use cases and orchestration"
echo "    src/infrastructure/    — External systems"
echo "    src/interfaces/        — APIs, CLI"
echo "    tests/                 — Unit and integration tests"
echo ""
echo "  Next steps:"
echo "    cd $TARGET_DIR"
echo "    uv sync"
echo "    claude"
