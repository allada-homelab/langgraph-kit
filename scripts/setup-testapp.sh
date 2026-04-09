#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$0")
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TESTAPP="$ROOT/testapp"

if [ -d "$TESTAPP" ]; then
  echo "testapp/ already exists. Remove it first to regenerate:"
  echo "  rm -rf $TESTAPP"
  exit 1
fi

echo "Generating test app from python-template..."
copier copy --defaults --data-file "$ROOT/.copier-testapp.yml" \
  gh:allada-homelab/python-template "$TESTAPP" --trust

echo "Overriding langgraph-kit to use local source..."
cd "$TESTAPP"
cat >> pyproject.toml <<'EOF'

[tool.uv.sources]
langgraph-kit = { path = "..", editable = true }
EOF

uv sync

echo ""
echo "testapp ready at $TESTAPP"
echo "Run tests: cd $TESTAPP && uv run pytest backend/"
