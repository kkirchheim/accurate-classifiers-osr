#!/usr/bin/env bash
# LOGFILE RUNS and WORKERS have to be defined

run_experiment() {
  CMD="python src/train.py --runs $RUNS --workers $WORKERS --test --log ${LOGFILE} ${1} ${2}"
  echo "> $CMD"
  $CMD > /dev/null 2>/dev/null
  echo "Returned: $?"
}

validate_config() {
  CMD="python src/validate.py --log - ${1} ${2}"
  echo "> $CMD"
  $CMD # > /dev/null 2>/dev/null
  echo "Returned: $?"
}
