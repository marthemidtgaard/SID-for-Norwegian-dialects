import os
import subprocess
from merge_metrics_over_seeds import dialect_eval_files

def merge_weather_slots(input_path, output_path):
    target_slots = {"weather/attribute", "condition_description", "condition_temperature"}

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w") as outfile:
        for line in infile:
            line = line.rstrip()

            if not line or line.startswith("#"):
                outfile.write(line + "\n")
                continue

            columns = line.split("\t")
            token, intent, slot = columns[1], columns[2], columns[3]

            if slot != "O":
                prefix, label = slot.split("-", 1)
                if label in target_slots:
                    slot = f"{prefix}-weather"

            # Write updated line
            columns[3] = slot
            outfile.write("\t".join(columns) + "\n")


def fix_files(list_of_models, output_dir):
    for model in list_of_models:
        dir_path = f'../machamp/logs/{model}/predictions'

        model_output_dir = os.path.join(output_dir, model)
        os.makedirs(model_output_dir, exist_ok=True)  # Create subfolder for each model

        for filename in os.listdir(dir_path):

            if filename.endswith(".conll"):
                filepath = os.path.join(dir_path, filename)
                output_path = os.path.join(model_output_dir, filename)
                merge_weather_slots(filepath, output_path)


def get_dialect_metrics(models_dir):
    for model_dir in os.listdir(models_dir):
        dirpath = os.path.join(models_dir, model_dir)

        for filename in os.listdir(dirpath):
            prediction_file = os.path.join(dirpath, filename)
            out_file = os.path.join(dirpath, f"{filename}.eval")

            command = ['python3',
                    'evalDialect.py',
                    'data/nomusic_test_weather.conll',
                    prediction_file]
        
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            with open(out_file, "w", encoding="utf-8") as out_f:
                    out_f.write(result.stdout)
        
        
def merge_metrics(models_dir):
    for model_dir in os.listdir(models_dir):
        dirpath = os.path.join(models_dir, model_dir)

        dialect_eval_files(dirpath, f'{models_dir}/{model_dir}.conll.dialect.eval')



#merge_weather_slots('data/nomusic_test.conll', 'data/nomusic_test_weather.conll')

#fix_files(['en_mbert', 'en_nb_rt1_noise_swap_xlmr', 'en_nb_rt1_noise_swap_mas_xlmr', 'nomusic_en_mas1_xlmr', 'nomusic_en_mas1_xlmr_int_mlm.1', 'nomusic_en_mas1_xlmr_two_joint_dep_pos.2', 'full_nomusic_en_mas1_xlmr'], 'predictions/test_set/weather')

#get_dialect_metrics('predictions/test_set/weather')

#merge_metrics('predictions/test_set/weather')

