#!/usr/bin/env bash
# End-to-end API demo with curl: high-value dispute -> supervisor approval -> caller notified,
# then a lost helpdesk response that does not create a duplicate ticket.
# Run it on a fresh database. On a second run the dispute already exists, so the agent
# refuses it before asking for confirmation and nothing reaches the supervisor queue
# (the correct outcome; the script then skips the approval step).
#
#   BASE=http://localhost:8000 SUPERVISOR_API_KEY=... ./examples/curl-demo.sh
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"
KEY=(); [[ -n "${SUPERVISOR_API_KEY:-}" ]] && KEY=(-H "X-API-Key: ${SUPERVISOR_API_KEY}")
JSON=(-H "content-type: application/json")
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval('d' + sys.argv[1]))" "$1"; }
say() { # conversation_id, text
  curl -sf "${JSON[@]}" -X POST "$BASE/v1/conversations/$1/messages" -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$2")"
}
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

step "health"
curl -sf "$BASE/health" | field '["llm_provider"], d["stt_provider"], d["tts_provider"], d["supervisor_auth"]'

step "new conversation"
CID=$(curl -sf "${JSON[@]}" -X POST "$BASE/v1/conversations" -d '{"language":"tr"}' | field '["id"]')
echo "$CID"

for line in "C-1002, TX-50015 işlemine itiraz ediyorum, ben yapmadım" "03.09.1975" "evet onaylıyorum"; do
  step "caller: $line"
  say "$CID" "$line" | field '["reply"]'
done

step "supervisor queue"
APR=$(curl -sf ${KEY[@]+"${KEY[@]}"} "$BASE/v1/approvals" | python3 -c "
import json, sys
mine = [a for a in json.load(sys.stdin) if a['conversation_id'] == sys.argv[1]]
print(mine[-1]['id'] if mine else '')" "$CID")
if [[ -n "$APR" ]]; then
  echo "$APR"
  curl -sf ${KEY[@]+"${KEY[@]}"} "${JSON[@]}" -X POST "$BASE/v1/approvals/$APR/decision" -d '{"decision":"approve","reviewer":"sup.demo"}' \
    | field '["status"], d["result"].get("id") or d["result"].get("error")'
  step "caller: Teşekkürler (hears the decision)"
  say "$CID" "Teşekkürler" | field '["reply"]'
else
  echo "nothing queued for this conversation (the charge already has an open dispute)"
fi

step "fault: helpdesk loses its response after creating the ticket"
curl -sf ${KEY[@]+"${KEY[@]}"} "${JSON[@]}" -X POST "$BASE/v1/demo/faults" \
  -d '{"path_prefix":"/helpdesk/tickets","method":"POST","mode":"error_after_commit"}' | field '["id"]'
CID2=$(curl -sf "${JSON[@]}" -X POST "$BASE/v1/conversations" -d '{"language":"tr"}' | field '["id"]')
say "$CID2" "müşteri numaram C-1003" > /dev/null
say "$CID2" "Mobil uygulama sürekli hata veriyor, şikayet kaydı açın lütfen" \
  | field '["reply"], d["tool_calls"][0]["attempts"], d["tool_calls"][0]["replayed"]'
echo "traces: $BASE/v1/conversations/$CID2/traces"
