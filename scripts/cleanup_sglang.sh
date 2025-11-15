#!/bin/bash
# Cleanup script to stop all SGLang servers
# Usage: ./scripts/cleanup_sglang.sh

set -e

echo "============================================"
echo "SGLang Server Cleanup"
echo "============================================"

# Find all SGLang server processes
PIDS=$(pgrep -f 'sglang.launch_server' || true)

if [ -z "$PIDS" ]; then
    echo "✅ No SGLang servers running"
else
    echo "🛑 Stopping SGLang servers..."
    echo "   PIDs: $PIDS"
    
    # Kill processes
    pkill -f 'sglang.launch_server' || true
    
    # Wait a moment
    sleep 2
    
    # Force kill if still running
    REMAINING=$(pgrep -f 'sglang.launch_server' || true)
    if [ -n "$REMAINING" ]; then
        echo "   ⚠️  Force killing remaining processes: $REMAINING"
        pkill -9 -f 'sglang.launch_server' || true
    fi
    
    echo "   ✅ All SGLang servers stopped"
fi

echo "============================================"
