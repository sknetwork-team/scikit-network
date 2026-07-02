#!/usr/bin/env bash
# Publish documentation gate for sknetwork-rs.
# Exits non-zero when the crate is not ready for crates.io publication.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Checking required documentation files"
for f in README.md AGENTS.md docs/rustdoc_style.md docs/PUBLISHING.md; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    exit 1
  fi
done

echo "==> Checking Cargo.toml publish metadata"
for field in description readme license repository authors rust-version; do
  if ! grep -q "^${field}" Cargo.toml; then
    echo "MISSING Cargo.toml field: ${field}"
    exit 1
  fi
done

echo "==> Building rustdoc"
cargo doc --no-deps --quiet

echo "==> Checking missing_docs (publish blocker)"
set +e
output=$(RUSTFLAGS='-D missing_docs' cargo doc --no-deps 2>&1)
status=$?
set -e

if [[ $status -ne 0 ]]; then
  count=$(echo "$output" | rg -c "missing documentation" || true)
  echo "FAIL: ${count:-?} public items lack rustdoc documentation."
  echo "See docs/PUBLISHING.md for remediation plan."
  echo "$output" | rg "missing documentation" | head -20
  exit 1
fi

echo "PASS: crate is documentation-complete for publish."
