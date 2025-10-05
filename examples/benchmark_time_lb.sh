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

# derive output CSV from program basename
PROG_BASENAME=$(basename -- "$PROGRAM")
SAFE_PROG=$(safe_tag "$PROG_BASENAME")
if [ -z "$SAFE_PROG" ]; then
  SAFE_PROG="program"
fi

OUT_CSV="${RESULTS_DIR}/results_time_lb_${SAFE_PROG}.csv"

# create CSV header if missing
if [ ! -f "$OUT_CSV" ]; then
  printf '%s\n' "timestamp,tasks,prog_n,algorithm,attempt,exit_code,time_s,per_task_times_s,per_task_pct,logfile,cmd" > "$OUT_CSV"
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

        # printable command string (trim trailing space)
        CMD_STR="$(printf '%q ' "${CMD_ARR[@]}")"
        CMD_STR="${CMD_STR% }"

        printf 'Running (attempt %d/%d): tasks=%s prog_n=%s algo=%s -> %s\n' "$ATT" "$REPEATS" "$TASKS" "$PROG_N" "$ALGO" "$LOGFILE"

        # execute and capture stdout+stderr
        "${CMD_ARR[@]}" > "$LOGFILE" 2>&1
        EXIT_CODE=$?

        # parse total time
        TIME_STR=$(grep -oE 'Done\. Time taken: [0-9]+(\.[0-9]+)?s' "$LOGFILE" | head -n1 | sed -E 's/Done\. Time taken: ([0-9]+(\.[0-9]+)?)s/\1/')
        if [ -z "$TIME_STR" ]; then
          TIME_STR="NA"
        fi

        # parse per-task effective times & percentages
        mapfile -t TASK_LINES < <(sed -n 's/Task \([0-9][0-9]*\): effective work time (excluding switches) = \([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)s (\([0-9][0-9]*\(\.[0-9][0-9]*\)\?\)% of total elapsed loop time).*/\1,\2,\4/p' "$LOGFILE" | sort -t, -k1,1n)

        if [ "${#TASK_LINES[@]}" -eq 0 ]; then
          PER_TASK_TIMES="NA"
          PER_TASK_PCTS="NA"
        else
          PER_TASK_TIMES=""
          PER_TASK_PCTS=""
          for line in "${TASK_LINES[@]}"; do
            IFS=, read -r tid tsecs tpct <<< "$line"
            if [ -z "$PER_TASK_TIMES" ]; then
              PER_TASK_TIMES="${tid}:${tsecs}"
              PER_TASK_PCTS="${tid}:${tpct}"
            else
              PER_TASK_TIMES="${PER_TASK_TIMES};${tid}:${tsecs}"
              PER_TASK_PCTS="${PER_TASK_PCTS};${tid}:${tpct}"
            fi
          done
        fi

        # append CSV row (only the cmd is quoted)
        printf '%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,"%s"\n' \
          "$TS" "$TASKS" "$PROG_N" "$ALGO" "$ATT" "$EXIT_CODE" "$TIME_STR" \
          "$PER_TASK_TIMES" "$PER_TASK_PCTS" "$LOGFILE" "$CMD_STR" >> "$OUT_CSV"

      done
    done
  done
done

echo "All runs completed. Results CSV: $OUT_CSV    Logs: $LOG_DIR/"
