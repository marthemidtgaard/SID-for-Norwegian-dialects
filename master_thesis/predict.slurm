#!/bin/bash
#SBATCH --job-name=sid_pred
#SBATCH --account=ec30
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=accel
#SBATCH --cpus-per-task=12
#SBATCH --qos=devel
#SBATCH --gpus=1
#SBATCH --exclude=gpu-14,gpu-4


source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8

#module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8
#module load nlpl-torchtext/0.15.2-foss-2022b-Python-3.10.8
#module load nlpl-transformers/4.43.4-foss-2022b-Python-3.10.8

MODEL_NAME_1=$1   #EX: nb_norbert
#MODEL_NAME_2=$2
#MODEL_NAME_3=$3
SPLIT=$2          #dev or test

if [ "$SPLIT" == "dev" ]; then
    DATA_PATH="data/nomusic_dev.conll"
elif [ "$SPLIT" == "new_dev" ]; then
    DATA_PATH="data/nomusic_new_dev.conll"
else
    DATA_PATH="data/nomusic_test.conll"
fi


#TODO for test set: Consolidate weather slots into 1

#for MODEL_NAME in "$MODEL_NAME_1" "$MODEL_NAME_2" "$MODEL_NAME_3"; do
for MODEL_NAME in "$MODEL_NAME_1"; do
    echo "Processing model: $MODEL_NAME"

    #Iterate over all 3 runs and predict for each run
    FULL_MODEL_PATH="../machamp/logs/$MODEL_NAME"

    PRED_DIR="../machamp/logs/$MODEL_NAME/predictions"
    mkdir -p $PRED_DIR

    while IFS= read -r dir; do
        SUBFOLDER_NAME=$(basename "$dir")

        if [[ "$SUBFOLDER_NAME" == "predictions" ]]; then
            continue
        fi

        MODEL="${FULL_MODEL_PATH}/${SUBFOLDER_NAME}/model.pt"
        PRED_PATH="${PRED_DIR}/${SUBFOLDER_NAME}.conll"

        python3 ../machamp/predict.py $MODEL $DATA_PATH $PRED_PATH #--dataset SID4LR
        python3 evalDialect.py data/nomusic_$SPLIT.conll $PRED_PATH > "$PRED_PATH".dialect.eval
        python3 data_processing/merge_metrics_over_seeds.py $PRED_DIR "predictions/${MODEL_NAME}.conll.eval" "predictions/${MODEL_NAME}.conll.dialect.eval"

    done < <(find "$FULL_MODEL_PATH" -mindepth 1 -maxdepth 1 -type d)
done



