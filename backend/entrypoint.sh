#!/bin/sh
set -e

python -m app.seed --admin-only
if [ "${SEED_DEMO:-0}" = "1" ]; then
  python -m app.seed --if-empty
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
