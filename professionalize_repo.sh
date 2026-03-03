#!/usr/bin/env bash
set -euo pipefail

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Error: run this inside your git repo folder."
  exit 1
}

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Create folders (if missing)
mkdir -p src/agents src/sql demo diagrams docs

# Make sure GitHub shows the diagrams folder
[ -f diagrams/.gitkeep ] || : > diagrams/.gitkeep

# Create files only if missing (won’t overwrite)
[ -f src/sql/sql_loader.py ] || : > src/sql/sql_loader.py

[ -f src/agents/controller_agent.py ] || : > src/agents/controller_agent.py
[ -f src/agents/filter_agent.py ] || : > src/agents/filter_agent.py
[ -f src/agents/retriever_agent.py ] || : > src/agents/retriever_agent.py
[ -f src/agents/writer_agent.py ] || : > src/agents/writer_agent.py

[ -f demo/app.py ] || : > demo/app.py
[ -f docs/WEEKLY_PLAN.md ] || : > docs/WEEKLY_PLAN.md
[ -f .env.example ] || : > .env.example

echo "Done: added professional folders/files (no renames)."
echo "Now run: git add . && git commit -m \"Add professional repo structure\" && git push"
