#!/usr/bin/env bash
set -u

MPIRUN_BIN=mpirun
PROGRAM="./md-lb"
COMMON_ARGS_ARRAY=("-o")
LOG_DIR="./logs"
RESULTS_DIR="results"
TASKS_LIST=(2 4 8 16 32 64)
PROG_N_LIST=(100 250 500 1000)
ALGO_LIST=(rcb rcbincr hilbert gilbert)
REPEATS=2

# program override via single positional argument
if [ $# -ge 1 ]; then
  PROGRAM="$1"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

safe_tag() {
  local s
  s=$(printf '%s' "$1" | tr ' /' '__')
  printf '%s' "$s" | LC_ALL=C tr -cd 'A-Za-z0-9_.-'
}

# escape field for CSV
escape_csv() {
  local s="$1"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

# derive output CSV from program basename
PROG_BASENAME=$(basename -- "$PROGRAM")
SAFE_PROG=$(safe_tag "$PROG_BASENAME")
if [ -z "$SAFE_PROG" ]; then
  SAFE_PROG="program"
fi

OUT_CSV="${RESULTS_DIR}/results_time_lb_${SAFE_PROG}.csv"

# create CSV header if missing
if [ ! -f "$OUT_CSV" ]; then
  printf '%s\n' "timestamp,tasks,prog_n,algorithm,attempt,exit_code,time_s,logfile,cmd" > "$OUT_CSV"
fi

for TASKS in "${TASKS_LIST[@]}"; do
  for PROG_N in "${PROG_N_LIST[@]}"; do
    for ALGO in "${ALGO_LIST[@]}"; do

      for (( ATT=1; ATT<=REPEATS; ATT++ )); do
        TS=$(date -Iseconds)
        TAG="t${TASKS}_n${PROG_N}_a${ALGO}_att${ATT}_$(date +%s)_$RANDOM"
        LOGFILE="${LOG_DIR}/run_${TAG}.log"

        # build the command array
        CMD_ARR=( "$MPIRUN_BIN" -n "$TASKS" "$PROGRAM" )
        for a in "${COMMON_ARGS_ARRAY[@]}"; do CMD_ARR+=( "$a" ); done
        CMD_ARR+=( -n "$PROG_N" -a "$ALGO" )

        # printable command string for CSV
        CMD_STR="$(printf '%q ' "${CMD_ARR[@]}")"

        printf 'Running (attempt %d/%d): tasks=%s prog_n=%s algo=%s -> %s\n' "$ATT" "$REPEATS" "$TASKS" "$PROG_N" "$ALGO" "$LOGFILE"

        # execute and capture stdout+stderr
        "${CMD_ARR[@]}" > "$LOGFILE" 2>&1
        EXIT_CODE=$?

        # parse time, expect: "Done. Time taken: XX.XXs"
        TIME_STR=$(grep -oE 'Done\. Time taken: [0-9]+(\.[0-9]+)?s' "$LOGFILE" | head -n1 | sed -E 's/Done\. Time taken: ([0-9]+(\.[0-9]+)?)s/\1/')

        if [ -z "$TIME_STR" ]; then
          TIME_STR="NA"
        fi

        # append CSV line (quote fields where appropriate)
        printf '%s,%s,%s,%s,%d,%d,%s,%s,%s\n' \
          "$(escape_csv "$TS")" \
          "$TASKS" \
          "$PROG_N" \
          "$(escape_csv "$ALGO")" \
          "$ATT" \
          "$EXIT_CODE" \
          "$(escape_csv "$TIME_STR")" \
          "$(escape_csv "$LOGFILE")" \
          "$(escape_csv "$CMD_STR")" >> "$OUT_CSV"

      done

    done
  done
done

echo "All runs completed. Results CSV: $OUT_CSV    Logs: $LOG_DIR/"
