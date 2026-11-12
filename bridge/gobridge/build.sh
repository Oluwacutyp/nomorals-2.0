#!/bin/sh
# Build both committed Go bridge binaries with build identity stamped.
#
# The status payload (--version too) carries commit + built time +
# the unwrap_inbound capability flag, so the supervisor — and the
# owner, via /pair and /health — can prove WHICH binary is actually
# running.  A binary built without this script reports commit "dev"
# / built "unknown": never a real release.
#
# Run from this directory (bridge/gobridge).  The stamp names the
# current HEAD: commit the code first, then run this, then commit
# the rebuilt binaries with the code (one commit).
set -eu
cd "$(dirname "$0")"
if ! command -v go >/dev/null 2>&1 \
    && [ -x "$HOME/workspace/tools/go/bin/go" ]; then
    PATH="$HOME/workspace/tools/go/bin:$PATH"
fi
COMMIT=$(git rev-parse --short HEAD)
BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FLAGS="-X main.bridgeCommit=$COMMIT -X main.bridgeBuilt=$BUILT"
export CGO_ENABLED=0
export GOPROXY=off
echo "building whatsapp-bridge (commit $COMMIT, built $BUILT)"
go build -trimpath -ldflags "$FLAGS" -o ../bin/whatsapp-bridge-linux .
GOOS=windows GOARCH=amd64 \
    go build -trimpath -ldflags "$FLAGS" -o ../bin/whatsapp-bridge.exe .
echo "binaries:"
ls -l ../bin/whatsapp-bridge-linux ../bin/whatsapp-bridge.exe
../bin/whatsapp-bridge-linux --version
