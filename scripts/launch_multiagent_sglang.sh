#!/bin/bash
# Launch multiple SGLang servers for multi-agent training
# Usage: ./scripts/launch_multiagent_sglang.sh [n_agents] [base_port]

set -e

N_AGENTS=${1:-2}
BASE_PORT=${2:-17987}
MODEL_PATH=${MODEL_PATH:-"/path/to/your/base/model"}
BASE_GPU_ID=${BASE_GPU_ID:-6}
# ✅ 可选：控制显存占用（默认 0.9）
MEM_FRACTION=${MEM_FRACTION:-0.9}
# ✅ LoRA 配置（必须与训练时的配置一致）
MAX_LORA_RANK=${MAX_LORA_RANK:-32}
# SGLang accepts: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, qkv_proj, gate_up_proj, all
# Use "all" to match "all-linear" behavior in training config
# LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"all"}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj v_proj"}

echo "============================================"
echo "Multi-Agent SGLang Launcher"
echo "============================================"
echo "N_AGENTS: ${N_AGENTS}"
echo "BASE_PORT: ${BASE_PORT}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "BASE_GPU_ID: ${BASE_GPU_ID}"
echo "MEM_FRACTION: ${MEM_FRACTION}"
echo "MAX_LORA_RANK: ${MAX_LORA_RANK}"
echo "LORA_TARGET_MODULES: ${LORA_TARGET_MODULES}"
echo "============================================"

# Check if model path exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path does not exist: ${MODEL_PATH}"
    echo "Set MODEL_PATH environment variable or update script"
    exit 1
fi

# Create logs directory
LOGS_DIR="./logs/sglang_multiagent"
mkdir -p "${LOGS_DIR}"

# Kill existing SGLang servers on target ports (optional cleanup)
for agent_id in $(seq 0 $((N_AGENTS - 1))); do
    PORT=$((BASE_PORT + agent_id))
    if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port ${PORT} already in use, killing existing process..."
        lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
done

# Launch SGLang servers
for agent_id in $(seq 0 $((N_AGENTS - 1))); do
    PORT=$((BASE_PORT + agent_id))
    GPU_ID=$((BASE_GPU_ID + agent_id))
    LOG_FILE="${LOGS_DIR}/agent${agent_id}_port${PORT}.log"
    
    echo "🚀 Starting SGLang server for Agent ${agent_id}..."
    echo "   - Port: ${PORT}"
    echo "   - GPU: ${GPU_ID}"
    echo "   - Log: ${LOG_FILE}"
    
    # ✅ 添加 --enable-lora 及必需的 LoRA 配置参数
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --port ${PORT} \
        --host 0.0.0.0 \
        --trust-remote-code \
        --mem-fraction-static ${MEM_FRACTION} \
        --enable-lora \
        --max-loras-per-batch 1 \
        --max-lora-rank ${MAX_LORA_RANK} \
        --lora-target-modules ${LORA_TARGET_MODULES} \
        --max-running-requests 64 \
        --schedule-policy fcfs \
        --skip-tokenizer-init \
        > "${LOG_FILE}" 2>&1 &
    
    SERVER_PID=$!
    echo "   - PID: ${SERVER_PID}"
    
    # Wait for server to start (check health endpoint)
    echo "   ⏳ Waiting for server to be ready..."
    for i in {1..60}; do
        if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "   ✅ Agent ${agent_id} server ready on port ${PORT}"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "   ❌ Timeout waiting for Agent ${agent_id} server"
            echo "   📋 Check logs: ${LOG_FILE}"
            echo ""
            echo "Last 20 lines of log:"
            tail -20 "${LOG_FILE}"
            exit 1
        fi
        sleep 2
    done
    echo ""
done

echo "============================================"
echo "✅ All SGLang servers started successfully"
echo "============================================"
echo "Active servers:"
for agent_id in $(seq 0 $((N_AGENTS - 1))); do
    PORT=$((BASE_PORT + agent_id))
    GPU_ID=$((BASE_GPU_ID + agent_id))
    echo "  Agent ${agent_id}: http://localhost:${PORT} (GPU ${GPU_ID})"
done
echo ""
echo "📊 Monitor logs:"
echo "  tail -f ${LOGS_DIR}/agent*.log"
echo ""
echo "🔧 Test server health:"
echo "  curl http://localhost:${BASE_PORT}/health"
echo ""
echo "🛑 Stop all servers:"
echo "  pkill -f 'sglang.launch_server'"
echo "  # or: ./scripts/cleanup_sglang.sh"
echo "============================================"