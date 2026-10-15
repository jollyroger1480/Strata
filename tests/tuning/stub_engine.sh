#!/bin/sh
# A stub engine for exercising `strata tune run` end to end.
#
# It speaks the real serve protocol (READY / PP / T / DONE) and makes its decode speed depend on one
# flag, so a sweep has something real to find. It is NOT a model and its numbers mean nothing about
# any hardware -- it exists so the *sweep machinery* can be tested without weights.
#
#   --kv int8   decodes at 200 tok/s
#   anything else (including --kv fp16) decodes at 100 tok/s
#
# The prompt's length comes from the GEN line's token list, so the workload is genuinely the one the
# caller asked for.
KV=unknown
for arg in "$@"; do
  if [ "$prev" = "--kv" ]; then KV="$arg"; fi
  prev="$arg"
done

echo "READY 32768 stop"
while read -r line; do
  case "$line" in
    QUIT|"") exit 0 ;;
    GEN*)
      ids=${line##* }
      count=$(printf '%s' "$ids" | tr ',' '\n' | wc -l | tr -d ' ')
      maxnew=$(printf '%s' "$line" | awk '{print $2}')
      [ -n "$maxnew" ] || maxnew=8
      printf 'PP %s %s 200.0 2560.0\n' "$count" "$count"
      if [ "$KV" = "int8" ]; then decode_ms=50.0; else decode_ms=100.0; fi
      # One T per token: the harness cross-checks the stream against DONE.
      i=0
      while [ "$i" -lt "$maxnew" ]; do
        echo "T $i"
        i=$((i + 1))
      done
      printf 'DONE %s %s 200.0 %s stop 0 0\n' "$maxnew" "$count" "$decode_ms"
      ;;
  esac
done
