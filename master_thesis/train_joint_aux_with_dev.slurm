#!/bin/bash
#SBATCH --job-name=sid
#SBATCH --account=ec30
#SBATCH --time=15:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=18G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=gpu-14

source ${HOME}/.bashrc

# Exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset


# Load necessary modules
module purge
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8


MODEL_NAME=$1
TRAIN_PATH=$2
TRAIN_SET=$(basename "$TRAIN_PATH")
AUX_TASK=$3


# Config files
AUX_TASK_CONFIG="../master_thesis/configs/${AUX_TASK}_aux.json"
CONFIG_FILE_MODEL="../master_thesis/configs/${MODEL_NAME}.json"


# Create a temporary JSON config file for MaChamp based on the training dataset passed
CONFIG_FILE_SID="../master_thesis/configs/${TRAIN_SET}.json"
cat > $CONFIG_FILE_SID <<EOL
{
    "SID4LR": {
        "train_data_path": "../master_thesis/data/${TRAIN_PATH}.conll",
        "dev_data_path": "../master_thesis/data/nomusic_new_dev.conll",
        "word_idx": 1,
        "tasks": {
            "slots": {
                "task_type": "seq_bio",
                "column_idx": 3
            },
            "intent": {
                "task_type": "classification",
                "column_idx": -1
            }
        }
    }
}
EOL


#Rename the output file
NEW_OUTPUT_FILE="${SLURM_JOB_ID}_${TRAIN_SET}_${MODEL_NAME}_joint_${AUX_TASK}.out"
mv slurm-${SLURM_JOB_ID}.out "$NEW_OUTPUT_FILE"


# Run the MaChamp training
cd ../machamp
for SEED in 1234 5678 8446; do
    python3 train.py --dataset_configs $CONFIG_FILE_SID $AUX_TASK_CONFIG1 $AUX_TASK_CONFIG2 \
    --parameters_config $CONFIG_FILE_MODEL \
    --seed "$SEED" \
    --name "${TRAIN_SET}_${MODEL_NAME}_joint_${AUX_TASK}" 
done

