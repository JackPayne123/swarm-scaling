# Source before running evals: `source scripts/env.sh`. Reads keys from ~/.pa-oauth at runtime so no
# secret is copied into the project.
K="$HOME/.pa-oauth"
export OPENAI_API_KEY="$(cat "$K/openai_key")"
export ANTHROPIC_API_KEY="$(cat "$K/anthropic_api_key")"
export GEMINI_API_KEY="$(cat "$K/gemini_key")"
export GOOGLE_API_KEY="$GEMINI_API_KEY"
# GLM via Z.ai's OpenAI-compatible endpoint: Inspect model string openai-api/zai/glm-5.3-flash
export ZAI_API_KEY="$(cat "$K/zai_key")"
export ZAI_BASE_URL="https://api.z.ai/api/paas/v4"
export OPENROUTER_API_KEY="$(cat "$K/openrouter_key")"
