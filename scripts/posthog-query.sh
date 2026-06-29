#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env.posthog"

# Load from .env.posthog if it exists, otherwise rely on global env vars
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi

if [[ -z "${POSTHOG_PERSONAL_API_KEY:-}" || -z "${POSTHOG_PROJECT_ID:-}" || -z "${POSTHOG_HOST:-}" ]]; then
  echo "ERROR: Set POSTHOG_PERSONAL_API_KEY, POSTHOG_PROJECT_ID, POSTHOG_HOST in ~/.zshrc or .env.posthog" >&2
  exit 1
fi

POSTHOG_API_HOST="${POSTHOG_API_HOST:-https://eu.posthog.com}"

usage() {
  cat << 'HELP'
Usage: posthog-query.sh <command> [args]

Commands:
  events <distinct_id> [limit]   List events for a distinct_id (default limit: 50)
  recent [limit]                 List most recent events across all distinct_ids (default: 30)
  search <event_name> [limit]    Search events by name (default: 30)
  count <distinct_id>            Count events for a distinct_id
  session <session_id>           List all events for a session (prefix session: auto-added)
  tail                           Stream latest events (polls every 5s)
  help                           Show this help

Environment:
  Reads .env.posthog from project root for credentials.

Examples:
  posthog-query.sh events debug-posthog-test 10
  posthog-query.sh session 02732348-1350-413a-9f0c-d6f47210b417
  posthog-query.sh search speech_continues 20
  posthog-query.sh recent 50
HELP
}

CMD="${1:-help}"
shift || true

case "$CMD" in
  events)
    DISTINCT_ID="${1:?Usage: posthog-query.sh events <distinct_id> [limit]}"
    LIMIT="${2:-50}"
    QUERY="SELECT timestamp, event, distinct_id FROM events WHERE distinct_id = '${DISTINCT_ID}' ORDER BY timestamp DESC LIMIT ${LIMIT}"
    ;;
  session)
    SID="${1:?Usage: posthog-query.sh session <session_id>}"
    [[ "$SID" != session:* ]] && SID="session:${SID}"
    LIMIT="${2:-100}"
    QUERY="SELECT timestamp, event, distinct_id FROM events WHERE distinct_id = '${SID}' ORDER BY timestamp ASC LIMIT ${LIMIT}"
    ;;
  recent)
    LIMIT="${1:-30}"
    QUERY="SELECT timestamp, event, distinct_id FROM events ORDER BY timestamp DESC LIMIT ${LIMIT}"
    ;;
  search)
    EVENT_NAME="${1:?Usage: posthog-query.sh search <event_name> [limit]}"
    LIMIT="${2:-30}"
    QUERY="SELECT timestamp, event, distinct_id FROM events WHERE event LIKE '%${EVENT_NAME}%' ORDER BY timestamp DESC LIMIT ${LIMIT}"
    ;;
  count)
    DISTINCT_ID="${1:?Usage: posthog-query.sh count <distinct_id>}"
    QUERY="SELECT event, count() as cnt FROM events WHERE distinct_id = '${DISTINCT_ID}' GROUP BY event ORDER BY cnt DESC"
    ;;
  tail)
    echo "Polling for latest events every 5s... (Ctrl+C to stop)"
    SEEN=""
    while true; do
      RESULT=$(curl -s "${POSTHOG_API_HOST}/api/projects/${POSTHOG_PROJECT_ID}/query/" \
        -H "Authorization: Bearer ${POSTHOG_PERSONAL_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\":{\"kind\":\"HogQLQuery\",\"query\":\"SELECT timestamp, event, distinct_id FROM events ORDER BY timestamp DESC LIMIT 20\"}}" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for r in data.get('results', []):
        key = r[0]+r[1]+r[2]
        print(f'{r[0]}  {r[1]:40s}  {r[2]}')
except: pass
" 2>/dev/null)
      if [[ -n "$RESULT" ]]; then
        echo "--- $(date -u +%H:%M:%S)Z ---"
        echo "$RESULT"
      fi
      sleep 5
    done
    ;;
  help|--help|-h)
    usage
    exit 0
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage
    exit 1
    ;;
esac

curl -s "${POSTHOG_API_HOST}/api/projects/${POSTHOG_PROJECT_ID}/query/" \
  -H "Authorization: Bearer ${POSTHOG_PERSONAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\":{\"kind\":\"HogQLQuery\",\"query\":\"${QUERY}\"}}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    rows = data.get('results', [])
    if not rows:
        print('No events found')
    else:
        for r in rows:
            print(f'{r[0]:28s} {r[1]:40s} {r[2]}' if len(r) >= 3 else str(r))
        print(f'\n({len(rows)} rows)')
except Exception as e:
    print(f'Error: {e}')
    print(sys.stdin.read() if hasattr(sys.stdin, 'read') else '')
" 2>&1
