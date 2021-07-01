#!/usr/bin/env bash
# We hypothesize that the OSR performance of a model correlates with its performance on the
# upstream task, i.e. predicting class membership. This script runs experiments of the same baseline model
# (softmax thresholding) with different encoders that are known to result in varying predictive performance.
# If our the hypothesis is true, we should observe that the OSR scores for the different encoders correlate with
# their accuracy

RUNS=28
WORKERS=28
ROOT="$(pwd)" # TODO: you might want to adjust this
LOGFILE="data/logs/vary-encoder-$(date +%Y-%m-%d-%H-%M-%S).log"
OVERRIDES="tool.paths.output=data/experiments/vary-encoder-$(date +%Y-%m-%d-%H-%M-%S) ${@}"

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")
echo "Loading helpers"
source "$SCRIPT_PATH/helpers.sh"


echo "Processing Image Data"
IMAGE_MODEL="softmax"
IMAGE_DATASETS="tiny-imagenet cifar-10 cifar-100 svhn"
IMAGE_ENCODERS="densenet-121 lenet-5 resnet-18 resnet-50"

for DATASET in ${IMAGE_DATASETS}
do
  for ENCODER in ${IMAGE_ENCODERS}
  do
    for MODEL in ${IMAGE_MODEL}
    do
      CONF="config/${DATASET}/${ENCODER}/${MODEL}.yaml"
      COMMENT="comment=features-$ENCODER"

      # when using a densenet, determine the appropriate pretrain-override
      if [ "${ENCODER}" = "densenet-121" ]
      then
        PRETRAINED="architecture.encoder.pretrained=${ROOT}/data/pretrained/densenet-121-imagenet-2010-openset"
        [ "${DATASET}" = "tiny-imagenet" ] && PRETRAINED="${PRETRAINED}-64x64.pt" || PRETRAINED="${PRETRAINED}-32x32.pt"
        run_experiment "${CONF}" "${OVERRIDES} ${COMMENT}-pretrained ${PRETRAINED}"
      fi

      run_experiment "${CONF}" "${OVERRIDES} ${COMMENT}"
      # validate_config "${CONF}" "${OVERRIDES}"
    done
  done
done

