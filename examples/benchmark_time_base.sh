#!/usr/bin/env bash
set -u

MPIRUN_BIN=mpirun
PROGRAM="./md"
COMMON_ARGS="-o"
LOG_DIR="./logs"
RESULTS_DIR="./results"

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

# derive name for results file
PROG_BASENAME=$(basename -- "$PROGRAM")
SAFE_PROG=$(safe_tag "$PROG_BASENAME")
if [ -z "$SAFE_PROG" ]; then
  SAFE_PROG="program"
fi

OUT_CSV="${RESULTS_DIR}/results_time_base_${SAFE_PROG}.csv"

# TASKS to test
TASKS_LIST=(2 4 8 16 32 64)

# CSV header
if [ ! -f "$OUT_CSV" ]; then
  printf '%s\n' "timestamp,tasks,exit_code,time_s,logfile,cmdline" > "$OUT_CSV"
fi

for TASKS in "${TASKS_LIST[@]}"; do
  for ATTEMPT in 1 2; do
    TS=$(date -Iseconds)
    TAG="t${TASKS}_a${ATTEMPT}_$(date +%s)_$RANDOM"
    LOGFILE="${LOG_DIR}/run_${TAG}.log"

    # build command as array to avoid word-splitting issues
    CMD=( "$MPIRUN_BIN" -n "$TASKS" "$PROGRAM" $COMMON_ARGS )

    echo "Running: program=${PROGRAM} tasks=${TASKS} attempt=${ATTEMPT} -> log=${LOGFILE}"
    "${CMD[@]}" > "$LOGFILE" 2>&1
    EXIT_CODE=$?

    # parse time from: 
    # Done. Time taken: XX.XXs
    TIME_STR=$(sed -n 's/.*Done\. Time taken: \([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)s.*/\1/p' "$LOGFILE" | head -n1)
    if [ -z "$TIME_STR" ]; then
      TIME_STR="NA"
    fi

    # serialize the full command line for traceability and append
    CMDLINE=$(printf '%s ' "${CMD[@]}")
    printf '%s,%s,%d,%s,%s,"%s"\n' "$TS" "$TASKS" "$EXIT_CODE" "$TIME_STR" "$LOGFILE" "$CMDLINE" >> "$OUT_CSV"
  done
done

echo "Done. Results: $OUT_CSV  Logs: $LOG_DIR/"
