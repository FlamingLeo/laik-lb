#!/usr/bin/env bash
set -u

MPIRUN_BIN=mpirun
PROGRAM=./md3d
COMMON_ARGS="-o"
OUT_CSV="results_md3d.csv"
LOG_DIR="./logs"

# TASKS to test
TASKS_LIST=(2 4 8 16 32 64)

mkdir -p "$LOG_DIR"

# CSV header (create if missing)
if [ ! -f "$OUT_CSV" ]; then
  printf '%s\n' "timestamp,tasks,exit_code,time_s,logfile,cmdline" > "$OUT_CSV"
fi

safe_tag() {
  # make a filename-friendly tag
  echo "$1" | tr ' /' '__' | tr -cd 'A-Za-z0-9_-.'
}

for TASKS in "${TASKS_LIST[@]}"; do
  TS=$(date -Iseconds)
  TAG="t${TASKS}_$(date +%s)"
  LOGFILE="${LOG_DIR}/run_${TAG}.log"

  # build command as array to avoid word-splitting issues
  CMD=( "$MPIRUN_BIN" -n "$TASKS" "$PROGRAM" $COMMON_ARGS )

  echo "Running: tasks=${TASKS}  -> log=${LOGFILE}"
  "${CMD[@]}" > "$LOGFILE" 2>&1
  EXIT_CODE=$?

  # parse time from the expected line
  # Done. Time taken: XX.XXs
  TIME_STR=$(sed -n 's/.*Done\. Time taken: \([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)s.*/\1/p' "$LOGFILE" | head -n1)
  if [ -z "$TIME_STR" ]; then
    TIME_STR="NA"
  fi

  # serialize the full command line for traceability
  CMDLINE=$(printf '%s ' "${CMD[@]}")
  # append CSV line
  printf '%s,%s,%d,%s,%s,"%s"\n' "$TS" "$TASKS" "$EXIT_CODE" "$TIME_STR" "$LOGFILE" "$CMDLINE" >> "$OUT_CSV"
done

echo "Done. Results: $OUT_CSV  Logs: $LOG_DIR/"
