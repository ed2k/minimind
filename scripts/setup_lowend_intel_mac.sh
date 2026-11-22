#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIN_DIR="$REPO_ROOT/selfchange/workspace/bin/llama.cpp"
GGUF_PATH="$REPO_ROOT/selfchange/workspace/promoted_models/gguf/champion.gguf"

echo "[setup] Checking Python & immutable kernel integrity (no PyTorch required)..."
python3 --version
python3 -m selfchange.kernel.supervisor --check-only

echo "[setup] Installing prebuilt llama.cpp binaries into $BIN_DIR..."
python3 -m selfchange.training.llamacpp_backend --install-bin --dest "$BIN_DIR"

if [[ -f "$GGUF_PATH" ]]; then
  echo "[setup] Running live GGUF smoke test on $GGUF_PATH..."
  python3 -m selfchange.training.llamacpp_backend --dest "$BIN_DIR" --smoke-test "$GGUF_PATH"
  echo
  echo "[ready] Low-end Intel Mac setup verified!"
  echo "  • Run 20-task strict eval locally: python3 -m selfchange.training.llamacpp_backend --eval-strict \"$GGUF_PATH\""
  echo "  • Run 1 local autonomous cycle: python3 -m selfchange.colab.colab_launcher --profile local_intel_cpu --require-model --eval-with-model --max-cycles 1"
  echo "  • Burst-train on Colab & refresh GGUF: python3 -m selfchange.colab.run_colab_cli --mode sync_train_gguf --overlay-local-code"
else
  echo
  echo "[setup] llama.cpp is installed, and kernel integrity passed."
  echo "[setup] No local GGUF champion found yet at: $GGUF_PATH"
  echo "[setup] Export/fetch a compact GGUF from Colab T4 with:"
  echo "  python3 -m selfchange.colab.run_colab_cli --mode export_gguf --overlay-local-code"
fi
