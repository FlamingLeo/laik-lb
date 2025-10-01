#!/usr/bin/env bash
set -u

MPIRUN_BIN=mpirun
PROGRAM=./md-lb
COMMON_ARGS_ARRAY=("-o")
OUT_CSV="results_md-lb.csv"
LOG_DIR="./logs"
TASKS_LIST=(2 4 8 16 32 64)
PROG_N_LIST=(100 250 500 1000)
ALGO_LIST=(rcb rcbincr hilbert gilbert)
REPEATS=2

mkdir -p "$LOG_DIR"

# create CSV header if missing
if [ ! -f "$OUT_CSV" ]; then
  printf '%s\n' "timestamp,tasks,prog_n,algorithm,attempt,exit_code,time_s,logfile,cmd" > "$OUT_CSV"
fi

# create safe tag
safe_tag() {
  echo "$1" | tr ' /' '__' | tr -cd 'A-Za-z0-9_-.'
}

# escape field for CSV
escape_csv() {
  local s="$1"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

for TASKS in "${TASKS_LIST[@]}"; do
  for PROG_N in "${PROG_N_LIST[@]}"; do
    for ALGO in "${ALGO_LIST[@]}"; do

      for (( ATT=1; ATT<=REPEATS; ATT++ )); do
        TS=$(date -Iseconds)
        TAG="t${TASKS}_n${PROG_N}_a${ALGO}_att${ATT}_$(date +%s)"
        LOGFILE="${LOG_DIR}/run_${TAG}.log"
        CMD_ARR=( "$MPIRUN_BIN" -n "$TASKS" "$PROGRAM" )
        for a in "${COMMON_ARGS_ARRAY[@]}"; do CMD_ARR+=( "$a" ); done
        CMD_ARR+=( -n "$PROG_N" -a "$ALGO" )

        # for CSV store a printable command string
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

        # append CSV line (quote fields)
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
