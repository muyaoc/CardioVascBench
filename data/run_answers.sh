#!/usr/bin/env bash
#set -euo pipefail

# Usage:
#   ./run_answers.sh apollo-7b
#   API_URL=http://127.0.0.1:8800 API_KEY=llmapi ./run_answers.sh ultramedical-8b

MODEL_KEY="${1:-apollo-7b}"

# ---------- API config ----------
API_URL_DEFAULT="http://223.109.239.65:30616"
API_KEY_DEFAULT="llmapi"
export API_URL="${API_URL:-$API_URL_DEFAULT}"
export API_KEY="${API_KEY:-$API_KEY_DEFAULT}"

# ---------- Path setup ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PATHS_CFG="$ROOT_DIR/config/paths.yml"
PY_SCRIPT="$SCRIPT_DIR/llms_answers_data_generator.py"

# ---------- Model map ----------
case "$MODEL_KEY" in
  qwen2.5-vl-7b-instruct)     MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/qwen2.5-vl-7b-instruct.yml" ;;
  qwen2.5-7b-instruct)        MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/qwen2.5-7b-instruct.yml" ;;
  qwen2.5-vl-32b-instruct)    MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/qwen2.5-vl-32b-instruct.yml" ;;



  pulse-7b)                   MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/pulse-7b.yml" ;;
  gem)                        MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/gem.yml" ;;
  ecg-chat)                   MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/ecg-chat.yml" ;;

  kimi-k2-0905)               MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/kimi-k2-0905.aihubmix.yml" ;;
  llama-4-maverick)           MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/llama-4-maverick.aihubmix.yml" ;;
  mimo-v2-flash)              MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/mimo-v2-flash.aihubmix.yml" ;;
  mistral-large-3)            MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/mistral-large-3.aihubmix.yml" ;;
  deepseek-v3.2)              MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/deepseek-v3.2.aihubmix.yml" ;;
  gpt-oss-120b)               MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/gpt-oss-120b.aihubmix.yml" ;;
  glm-4.7)                    MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/glm-4.7.aihubmix.yml" ;;


  llama-3-8b-instruct)        MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/llama-3-8b-instruct.yml" ;;
  llama-3-70b-instruct)       MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/llama-3-70b-instruct.yml" ;;
  llama-3-70b)                MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/llama-3-70b.yml" ;;
  gpt-oss-20b)                MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/gpt-oss-20b.aihubmix.yml" ;;

  gpt-4o)                     MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/gpt-4o.aihubmix.yml" ;;
  gpt-5.2)                    MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/gpt-5.2.aihubmix.yml" ;;
  grok-4)                     MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/grok-4.aihubmix.yml" ;;
  qwen3-max)                  MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/qwen3-max.aihubmix.yml" ;;
  claude-sonnet-4-5)          MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/claude-sonnet-4-5.aihubmix.yml" ;;
  gemini-3-pro-preview)       MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/gemini-3-pro-preview.aihubmix.yml" ;;

  meditron3-8b)               MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/meditron3-8b.yml" ;;
  lingshu-7b)                 MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/lingshu-7b.yml" ;;
  lingshu-32b)                MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/lingshu-32b.yml" ;;
  ii-medical-8b)              MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/ii-medical-8b.yml" ;;
  sunsimiao)                  MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/sunsimiao.yml" ;;
  apollo-7b)                  MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/apollo-7b.yml" ;;
  apollo-72b)                 MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/apollo-72b.yml" ;;
  apollo2-9b)                 MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/apollo2-9b.yml" ;;
  biomistral-7b)              MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/biomistral-7b.yml" ;;
  huatuogpt-7b)               MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/huatuogpt-7b.yml" ;;
  huatuogpt-8b)               MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/huatuogpt-8b.yml" ;;
  huatuogpt-70b)              MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/huatuogpt-70b.yml" ;;
  huatuogpt-72b)              MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/huatuogpt-72b.yml" ;;
  mistral-7b-instruct)        MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/mistral-7b-instruct.yml" ;;
  openbiollm-70b)             MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/openbiollm-70b.yml" ;;
  qwen3b-vl-4b-thinking)     MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/qwen3-vl-4b-thinking.yml" ;;
  shizhengpt-7b|shizhen-7b)   MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/shizhengpt-7b.yml" ;;
  shizhengpt-32b|shizhen-32b) MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/shizhengpt-32b.yml" ;;
  ultramedical-8b)            MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/ultramedical-8b.yml" ;;
  *) echo "Unknown model key: $MODEL_KEY";  ;;
#  medalpaca-7b)               MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/medalpaca-7b.yml" ;;
#  meditron-7b)                MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/meditron-7b.yml" ;;
#  ming-7b)                    MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/ming-7b.yml" ;;
#  openbiollm-8b)              MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/openbiollm-8b.yml" ;;
#  sunsimiao-qwen-7b)          MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/sunsimiao-qwen-7b.yml" ;
#  ultramedical-70b)           MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/ultramedical-70b.yml" ;;
#  ziya-llama-13b-med)         MODEL_CFG="$ROOT_DIR/config/llm_tpl_config/ziya-llama-13b-med.yml" ;;``
esac

# ---------- Sanity checks ----------
[[ -f "$MODEL_CFG" ]] || { echo "Missing model config: $MODEL_CFG";  exit 1; }
[[ -f "$PATHS_CFG" ]] || { echo "Missing paths config: $PATHS_CFG";  exit 1; }
[[ -f "$PY_SCRIPT" ]] || { echo "Missing python script: $PY_SCRIPT";  exit 1; }

# ---------- Run ----------
echo "== Running model: $MODEL_KEY"
echo "== MODEL_CFG: $MODEL_CFG"
echo "== PATHS_CFG: $PATHS_CFG"
echo "== API_URL:   $API_URL"

# --- ADD THIS LINE ---
# Change to the project root so all relative paths in configs work
cd "$ROOT_DIR" || { echo "Failed to cd to $ROOT_DIR"; exit 1; }
# ---------------------

python3 "$PY_SCRIPT" \
  --model-cfg "$MODEL_CFG" \
  --paths-cfg "$PATHS_CFG"