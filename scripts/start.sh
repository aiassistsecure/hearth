#!/usr/bin/env bash
# hearth — turn it all on, keep it on.
#
#   ./scripts/start.sh up        start the fleet, persistent (survives logout,
#                                restarts on crash, logs to $HEARTH_HOME/logs)
#   ./scripts/start.sh status    is it running, and what does the fleet say
#   ./scripts/start.sh logs      follow the gateway log
#   ./scripts/start.sh down      stop everything cleanly (SIGTERM -> spine
#                                records `unloaded`, children reaped)
#
# Configuration is a file, not flags, because a fleet definition is standing
# state and flags evaporate with the shell that typed them. First run writes a
# template to $HEARTH_HOME/fleet.conf and exits so you can fill it in.
#
# Every knob still works the flag > env > default way underneath — this script
# only *carries* your env (HEARTH_PARALLEL, HEARTH_GPU_LAYERS, HEARTH_PORT,
# HEARTH_MAX_INFLIGHT, HEARTH_CTX) into the persistent process.
#
# Written for bash 3.2 on purpose: that is what macOS ships, and a start
# script that needs a newer bash than the OS has is a support ticket.

set -u

HEARTH_HOME="${HEARTH_HOME:-$HOME/.hearth}"
CONF="$HEARTH_HOME/fleet.conf"
PIDFILE="$HEARTH_HOME/hearth.pid"
KEEPER_PIDFILE="$HEARTH_HOME/keeper.pid"
LOGDIR="$HEARTH_HOME/logs"
GATEWAY_LOG="$LOGDIR/gateway.log"
HEARTH_BIN="${HEARTH_BIN:-hearth}"

say()  { printf 'hearth: %s\n' "$*"; }
die()  { printf 'hearth: %s\n' "$*" >&2; exit 1; }

# A pid is "running" only if it exists AND is one of ours. PID reuse after a
# reboot would otherwise make status lie and down kill a stranger's process.
alive() {
  local pid="$1"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o command= 2>/dev/null | grep -q hearth
}

write_template() {
  mkdir -p "$HEARTH_HOME"
  cat > "$CONF" <<'EOF'
# hearth fleet — one model per line: NAME=[gguf:|safetensors:]PATH:GIB[@CTX]
# Declaration order IS priority order: first fit, never best fit.
# Lines starting with # are ignored.
#
# Existing GGUF syntax stays valid and uses llama.cpp:
# model muse=/models/muse.gguf:20
# model deepseek-r1:32b=gguf:/models/deepseek.gguf:20
#
# Native Hugging Face/Transformers packages use vLLM:
# model imagine-v8=safetensors:/models/Interchained_imagine-v8:3@16384
#
# @CTX overrides the fleet-wide `ctx` below FOR THAT MODEL ONLY. Use it when a
# fleet mixes context sizes: one number cannot be right for both a 1M-context
# model and an 8B, and the KV budget is computed per model from whichever
# applies.
#
# model kimi=/models/kimi.gguf:35@65536
# model gpt-oss:20b=/models/gpt-oss.gguf:12@32768

# The card, in GiB. hearth refuses at declare time what will not fit.
total_gib 24

# Gateway port (OpenAI-compatible). 11434 so Ollama clients need no change.
port 11434

# Slots per model (concurrent requests each model answers). 8 is the
# production default; each slot costs KV cache.
parallel 8

# Context per slot. SET THIS. KV cache is ctx * parallel, so leaving it at 0
# means "the model's full trained context" — 131072 on gpt-oss, 1048576 on
# Kimi-Linear — multiplied by every slot above. A fleet at `parallel 8` with
# no ctx once spent ~14 GiB on KV cache nothing had declared and took the card
# down. 16384 is a sane production start; raise it once /residency shows real
# headroom.
ctx 16384

# Requests the gateway will hold at once before answering 503. Default 64.
# max_inflight 64

# GPU layers: -1 = all (default). 0 = CPU only — choose it, never fall into it.
# gpu_layers -1

# Extra args passed to llama-server verbatim for GGUF models, appended LAST so
# they override hearth's defaults. Time-to-first-token knobs live here:
#   -ub 2048           bigger physical prefill batch (default 512)
#   --cache-reuse 256  reuse a cached KV prefix by shifting, not re-prefill
#   -fa on             flash attention (default auto)
#   --metrics          prometheus /metrics per child
# extra -ub 2048 --cache-reuse 256 -fa on --metrics
EOF
  say "wrote a template to $CONF — fill in your models and run '$0 up' again"
}

read_conf() {
  [ -f "$CONF" ] || { write_template; exit 0; }
  MODELS=(); TOTAL_GIB=24; PORT="${HEARTH_PORT:-11434}"
  PARALLEL="${HEARTH_PARALLEL:-8}"; GPU_LAYERS="${HEARTH_GPU_LAYERS:-}"
  CTX="${HEARTH_CTX:-}"; MAX_INFLIGHT="${HEARTH_MAX_INFLIGHT:-}"
  EXTRA=()
  while IFS= read -r line; do
    line="${line%%#*}"
    [ -z "${line// /}" ] && continue
    set -- $line
    case "$1" in
      model)        MODELS+=("$2") ;;
      total_gib)    TOTAL_GIB="$2" ;;
      port)         PORT="$2" ;;
      parallel)     PARALLEL="$2" ;;
      gpu_layers)   GPU_LAYERS="$2" ;;
      # ctx WAS MISSING, AND IT IS THE KNOB THAT CAUSED THE INCIDENT.
      # `parallel` was configurable here and `ctx` was not, but KV cache is
      # the PRODUCT of the two: on 2026-08-28 a fleet at `parallel 8` with no
      # explicit ctx spent ~14 GiB on KV cache nothing had declared and took
      # the card down. The operator could configure the multiplier and not the
      # multiplicand, and adding `ctx` to this file to fix it hit
      # "unknown directive 'ctx'" — so the documented remedy was unreachable
      # from the documented config surface.
      ctx)          CTX="$2" ;;
      max_inflight) MAX_INFLIGHT="$2" ;;
      extra)        shift; EXTRA+=("$@") ;;
      *)            die "fleet.conf: unknown directive '$1' (known: model, total_gib, port, parallel, ctx, max_inflight, gpu_layers, extra)" ;;
    esac
  done < "$CONF"
  [ "${#MODELS[@]}" -gt 0 ] || die "no models in $CONF — add 'model NAME=[gguf:|safetensors:]PATH:GIB[@CTX]' lines"
}

cmd_up() {
  command -v "$HEARTH_BIN" >/dev/null 2>&1 \
    || die "hearth is not on PATH — cargo install hearth-serve (or set HEARTH_BIN)"
  read_conf
  mkdir -p "$LOGDIR"

  if alive "$(cat "$KEEPER_PIDFILE" 2>/dev/null)"; then
    say "already running (keeper pid $(cat "$KEEPER_PIDFILE")) — '$0 status' to look"
    exit 0
  fi

  local args=(up --total-gib "$TOTAL_GIB" --port "$PORT" --parallel "$PARALLEL")
  [ -n "$GPU_LAYERS" ] && args+=(--gpu-layers "$GPU_LAYERS")
  # Passed as a FLAG, not left to the inherited env. `hearth up` reads
  # flag > env > default, and a fleet definition that only works because the
  # invoking shell happened to export something is not standing state.
  [ -n "$CTX" ] && args+=(--ctx "$CTX")
  [ -n "$MAX_INFLIGHT" ] && args+=(--max-inflight "$MAX_INFLIGHT")
  local m
  for m in "${MODELS[@]}"; do args+=(--model "$m"); done
  # `extra` goes to llama-server, and the ONLY way it gets there is after a
  # bare `--`. Appending it to hearth's own flags (what this line did before)
  # made `hearth up` silently ignore every token: fleet.conf said
  # `extra --jinja`, the children ran without it, and nothing said so.
  [ "${#EXTRA[@]}" -gt 0 ] && args+=(-- "${EXTRA[@]}")

  # The keeper: run the gateway, and if it CRASHES, restart it after a pause.
  # A clean exit (SIGTERM from `down`, exit 0) is respected — a supervisor
  # that resurrects what you deliberately stopped is a supervisor you turn off.
  # The pause is 5s so a model broken at boot cannot hot-loop the GPU;
  # the spine records every death either way (`hearth why <model>`).
  nohup bash -c '
    HEARTH_BIN="$1"; PIDFILE="$2"; LOG="$3"; shift 3
    while :; do
      "$HEARTH_BIN" "$@" >> "$LOG" 2>&1 &
      child=$!
      echo "$child" > "$PIDFILE"
      trap "kill -TERM $child 2>/dev/null; wait $child; rm -f \"$PIDFILE\"; exit 0" TERM INT
      wait "$child"; code=$?
      rm -f "$PIDFILE"
      trap - TERM INT
      if [ "$code" -eq 0 ] || [ "$code" -eq 143 ] || [ "$code" -eq 130 ]; then
        echo "hearth: exited cleanly ($code) — not restarting" >> "$LOG"
        exit 0
      fi
      echo "hearth: died with $code — restarting in 5s (see $LOG and: hearth why <model>)" >> "$LOG"
      sleep 5
    done
  ' keeper "$HEARTH_BIN" "$PIDFILE" "$GATEWAY_LOG" "${args[@]}" \
    >/dev/null 2>&1 &
  echo $! > "$KEEPER_PIDFILE"

  sleep 2
  if alive "$(cat "$PIDFILE" 2>/dev/null)"; then
    say "up — gateway on http://127.0.0.1:$PORT (pid $(cat "$PIDFILE"), keeper $(cat "$KEEPER_PIDFILE"))"
    say "logs: $GATEWAY_LOG   ·   '$0 status' for the fleet   ·   '$0 down' to stop"
  else
    say "did not come up — last log lines:"
    tail -5 "$GATEWAY_LOG" 2>/dev/null
    exit 1
  fi
}

cmd_status() {
  local keeper gw
  keeper="$(cat "$KEEPER_PIDFILE" 2>/dev/null)"; gw="$(cat "$PIDFILE" 2>/dev/null)"
  if alive "$gw"; then
    say "running — gateway pid $gw$( alive "$keeper" && printf ', keeper %s' "$keeper" )"
  elif alive "$keeper"; then
    say "keeper $keeper is up but the gateway is between restarts — check $GATEWAY_LOG"
  else
    say "not running"
  fi
  # The fleet's own answer beats anything this script can infer.
  "$HEARTH_BIN" status 2>/dev/null || true
}

cmd_logs() { exec tail -f "$GATEWAY_LOG"; }

cmd_down() {
  local keeper gw stopped=""
  keeper="$(cat "$KEEPER_PIDFILE" 2>/dev/null)"; gw="$(cat "$PIDFILE" 2>/dev/null)"
  # Keeper first, or it resurrects the gateway we are about to stop.
  if alive "$keeper"; then kill -TERM "$keeper" 2>/dev/null; stopped="keeper $keeper"; fi
  if alive "$gw"; then
    kill -TERM "$gw" 2>/dev/null   # SIGTERM: spine records unloaded, children reaped
    stopped="${stopped:+$stopped, }gateway $gw"
  fi
  [ -n "$stopped" ] || { say "nothing was running"; rm -f "$PIDFILE" "$KEEPER_PIDFILE"; exit 0; }
  # Give the gateway real time to shut down cleanly: SIGTERM triggers
  # record-unloaded + reap-children + flush, and killing that with -9 loses
  # the final events from the spine. 10s in 1s steps, force only after.
  local i=0
  while [ $i -lt 10 ] && alive "$gw"; do sleep 1; i=$((i+1)); done
  alive "$gw" && { say "gateway still up after ${i}s of TERM — forcing (the last events may be missing from the spine)"; kill -9 "$gw" 2>/dev/null; }
  alive "$keeper" && kill -9 "$keeper" 2>/dev/null
  rm -f "$PIDFILE" "$KEEPER_PIDFILE"
  say "stopped ($stopped) — the shutdown is in the history: hearth why <model>"
}

case "${1:-}" in
  up)     cmd_up ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  down)   cmd_down ;;
  *)      echo "usage: $0 up|status|logs|down    (config: $CONF)"; exit 2 ;;
esac
