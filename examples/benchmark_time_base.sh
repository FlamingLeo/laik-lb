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

# CSV header (now includes per-task columns)
if [ ! -f "$OUT_CSV" ]; then
  printf '%s\n' "timestamp,tasks,exit_code,time_s,per_task_times_s,per_task_pct,logfile,cmdline" > "$OUT_CSV"
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

    # parse total time from:
    # Done. Time taken: XX.XXs
    TIME_STR=$(sed -n 's/.*Done\. Time taken: \([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)s.*/\1/p' "$LOGFILE" | head -n1)
    if [ -z "$TIME_STR" ]; then
      TIME_STR="NA"
    fi

    # parse per-task effective times & percentages:
    # Task <id>: effective work time (excluding switches) = <seconds>s (<percent>% of total elapsed loop time)
    # capture id, seconds, percent, then sort by id
    mapfile -t TASK_LINES < <(sed -n 's/Task \([0-9][0-9]*\): effective work time (excluding switches) = \([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)s (\([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)% of total elapsed loop time).*/\1,\2,\4/p' "$LOGFILE" | sort -t, -k1,1n)

    if [ "${#TASK_LINES[@]}" -eq 0 ]; then
      PER_TASK_TIMES="NA"
      PER_TASK_PCTS="NA"
    else
      PER_TASK_TIMES=""
      PER_TASK_PCTS=""
      for line in "${TASK_LINES[@]}"; do
        IFS=, read -r tid tsecs tpct <<< "$line"
        
        # append entries in the form id:seconds and id:percent, separated by semicolons
        if [ -z "$PER_TASK_TIMES" ] || [ "$PER_TASK_TIMES" = "" ]; then
          PER_TASK_TIMES="${tid}:${tsecs}"
          PER_TASK_PCTS="${tid}:${tpct}"
        else
          PER_TASK_TIMES="${PER_TASK_TIMES};${tid}:${tsecs}"
          PER_TASK_PCTS="${PER_TASK_PCTS};${tid}:${tpct}"
        fi
      done
    fi

    # serialize the full command line for traceability and append
    CMDLINE=$(printf '%s ' "${CMD[@]}")

    # append CSV row with new per-task columns
    printf '%s,%s,%d,%s,%s,%s,%s,"%s"\n' \
      "$TS" "$TASKS" "$EXIT_CODE" "$TIME_STR" "$PER_TASK_TIMES" "$PER_TASK_PCTS" "$LOGFILE" "$CMDLINE" >> "$OUT_CSV"
  done
done

echo "Done. Results: $OUT_CSV  Logs: $LOG_DIR/"
