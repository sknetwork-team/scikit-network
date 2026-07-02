#!/usr/bin/env bash
# Verify the crates.io tarball contains only publishable artifacts.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Building publish package"
if ! list=$(cargo package --allow-dirty --list 2>&1); then
  echo "FAIL: cargo package --list failed"
  echo "$list"
  exit 1
fi

echo "==> Checking forbidden paths"
forbidden=(
  "benchmarking/"
  "AGENTS.md"
  "PORTING_MEMO.md"
  "__pycache__"
  ".bench-venv"
)
for pattern in "${forbidden[@]}"; do
  if echo "$list" | grep -E -q "$pattern"; then
    echo "FAIL: publish tarball contains forbidden path matching '$pattern'"
    echo "$list" | grep -E "$pattern" | head -10
    exit 1
  fi
done

echo "==> Checking required paths"
required=(
  "Cargo.toml"
  "README.md"
  "LICENSE"
  "src/lib.rs"
)
for path in "${required[@]}"; do
  if ! echo "$list" | grep -F -x -q "$path"; then
    echo "FAIL: publish tarball missing required path: $path"
    exit 1
  fi
done

echo "==> Checking keyword count (crates.io max: 5)"
keyword_line=$(grep -E '^keywords' Cargo.toml || true)
keyword_count=$(echo "$keyword_line" | grep -o '"[^"]\+"' | wc -l)
if [[ "$keyword_count" -gt 5 ]]; then
  echo "FAIL: Cargo.toml has $keyword_count keywords (max 5 on crates.io)"
  exit 1
fi

echo "PASS: publish tarball looks ready for crates.io."
