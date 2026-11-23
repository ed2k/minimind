#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LABEL="${MINIMIND_LAUNCHD_LABEL:-com.minimind.selfchange}"
WORKSPACE_REL="${MINIMIND_WORKSPACE_DIR:-selfchange/workspace/local_train}"
MODEL_REL="${MINIMIND_MODEL_PATH:-selfchange/workspace/promoted_models/gguf/0_minimal_cpu_champion-q4_k_m.gguf}"
TRAIN_BASE_REL="${MINIMIND_TRAIN_BASE_PTH:-out/minimal_cpu_champion.pth}"
CHUNK_CYCLES="${MINIMIND_CHUNK_CYCLES:-4}"
COOLDOWN_SEC="${MINIMIND_COOLDOWN_SEC:-20}"

WORKSPACE_ABS="$REPO_ROOT/$WORKSPACE_REL"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/${LABEL}.plist"
UID_NUM="$(id -u)"

mkdir -p "$PLIST_DIR" "$WORKSPACE_ABS"
rm -f "$WORKSPACE_ABS/STOP"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string>
    <string>-i</string>
    <string>${REPO_ROOT}/selfchange/workspace/run_intel_mac.sh</string>
    <string>python</string>
    <string>-u</string>
    <string>-m</string>
    <string>selfchange.colab.forever_daemon</string>
    <string>--profile</string>
    <string>local_intel_cpu</string>
    <string>--workspace-dir</string>
    <string>${WORKSPACE_REL}</string>
    <string>--model-path</string>
    <string>${MODEL_REL}</string>
    <string>--train-base-pth</string>
    <string>${TRAIN_BASE_REL}</string>
    <string>--autonomy-level</string>
    <string>1</string>
    <string>--train-every-n-cycles</string>
    <string>2</string>
    <string>--chunk-cycles</string>
    <string>${CHUNK_CYCLES}</string>
    <string>--local-sft-count</string>
    <string>48</string>
    <string>--local-dpo-count</string>
    <string>16</string>
    <string>--local-train-timeout</string>
    <string>1200</string>
    <string>--local-cpu-threads</string>
    <string>2</string>
    <string>--cooldown-sec</string>
    <string>${COOLDOWN_SEC}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>Nice</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>${WORKSPACE_ABS}/launchd.log</string>
  <key>StandardErrorPath</key>
  <string>${WORKSPACE_ABS}/launchd.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/${UID_NUM}" "$PLIST_PATH" >/dev/null 2>&1 || launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
if ! launchctl bootstrap "gui/${UID_NUM}" "$PLIST_PATH" >/dev/null 2>&1; then
  launchctl load -w "$PLIST_PATH"
fi

echo "Installed and started $LABEL ($PLIST_PATH)"
echo "Check status : cat $WORKSPACE_ABS/DAEMON_STATUS.json"
echo "Graceful stop: touch $WORKSPACE_ABS/STOP"
echo "Unload agent : launchctl bootout gui/${UID_NUM} $PLIST_PATH"
