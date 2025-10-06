#!/usr/bin/env bash
set -u

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

# THREADS to test
THREADS_LIST=(2 4 8 16 32 64)

# CSV header
if [ ! -f "$OUT_CSV" ]; then
  printf '%s\n' "timestamp,threads,exit_code,time_s,logfile,cmdline" > "$OUT_CSV"
fi

for THREADS in "${THREADS_LIST[@]}"; do
  for ATTEMPT in 1 2; do
    TS=$(date -Iseconds)
    TAG="th${THREADS}_a${ATTEMPT}_$(date +%s)_$RANDOM"
    LOGFILE="${LOG_DIR}/run_${TAG}.log"

    # set OpenMP threads
    export OMP_NUM_THREADS="$THREADS"

    # build command as array
    CMD=( "$PROGRAM" $COMMON_ARGS )

    echo "Running: program=${PROGRAM} threads=${THREADS} attempt=${ATTEMPT} -> log=${LOGFILE}"
    "${CMD[@]}" > "$LOGFILE" 2>&1
    EXIT_CODE=$?

    # parse total time from log: Time taken: XX.XXs
    TIME_STR=$(sed -n 's/.*Time taken: \([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)s.*/\1/p' "$LOGFILE" | head -n1)
    if [ -z "$TIME_STR" ]; then
      TIME_STR="NA"
    fi

    # serialize full command line
    CMDLINE=$(printf '%s ' "${CMD[@]}")

    # append CSV row
    printf '%s,%s,%d,%s,%s,"%s"\n' \
      "$TS" "$THREADS" "$EXIT_CODE" "$TIME_STR" "$LOGFILE" "$CMDLINE" >> "$OUT_CSV"
  done
done

echo "Done. Results: $OUT_CSV  Logs: $LOG_DIR/"
