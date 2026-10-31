#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BACKUP_DIR="${BACKUP_DIR:-$BASE/backups}"
mkdir -p "$BACKUP_DIR"

cd "$BASE/deploy"
STAMP=$(date +%F_%H%M%S)
OUT="$BACKUP_DIR/weibo_$STAMP.sql.gz"

docker compose exec -T db pg_dump -U weibo weibo </dev/null | gzip > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"

# 保留 14 天
find "$BACKUP_DIR" -name 'weibo_*.sql.gz' -mtime +14 -delete

echo "$(date -Is) backup ok $OUT $(du -h "$OUT" | cut -f1)"
