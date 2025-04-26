import os
import json
import sys
import numpy as np
import pandas as pd

def eval_files(input_folder, output_file):
    metrics = []

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        if filename.endswith(".conll.eval"):
            with open(filepath, "r") as f:
                data = json.load(f)
                
                metrics_entry = {
                    "file": filename,
                    "intent_accuracy": data["intent"]["accuracy"]["accuracy"],
                    "slot_precision": data["slots"]["span_f1"]["precision"],
                    "slot_recall": data["slots"]["span_f1"]["recall"],
                    "slot_f1": data["slots"]["span_f1"]["span_f1"],
                    "sum": data["sum"]
                }
            
                metrics.append(metrics_entry)
    
    eval_df = pd.DataFrame(metrics)
    eval_summary = eval_df.drop(columns=["file"]).agg(["mean", "std"])
    
    eval_summary_dict = eval_summary.round(3).to_dict()

    with open(output_file, "w") as json_file:
        json.dump(eval_summary_dict, json_file, indent=4)


def dialect_eval_files(input_folder, output_file):
    metrics = {}

    # Read all .dialect.eval files
    for filename in os.listdir(input_folder):
        if filename.endswith(".conll.dialect.eval"):
            filepath = os.path.join(input_folder, filename)

            with open(filepath, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) > 1 and parts[0] not in ["class:"]:
                    metric = " ".join(parts[:-5])
                    values = list(map(float, parts[-5:]))

                    if metric not in metrics:
                        metrics[metric] = []
                    
                    metrics[metric].append(values)
    

    # Compute mean and std for each metric
    aggregated_metrics = {}
    for key, value_list in metrics.items():
        values_array = np.array(value_list)  # Convert to NumPy array for easy calculation
        mean_values = np.mean(values_array, axis=0)
        std_values = np.std(values_array, axis=0)

        # Store mean with std in parentheses
        aggregated_metrics[key] = [
            f"{mean:.3f} ({std:.3f})" for mean, std in zip(mean_values, std_values)
        ]


    grouped_categories = [
        ["slot recall:", "slot precision:", "slot f1:"],
        ["intent accuracy:", "fully correct:"],  # These stay together
        ["unlabeled slot recall:", "unlabeled slot precision:", "unlabeled slot f1:"],
        ["loose slot recall:", "loose slot precision:", "loose slot f1:"]
    ]
    

    # Write results in the original table format
    with open(output_file, "w") as f:
        f.write("class:                    B               N               T               V               all\n\n")

        for group in grouped_categories:
            for key in group:
                if key in aggregated_metrics:
                    line = f"{key:<25} " + "  ".join(aggregated_metrics[key]) + "\n"
                    f.write(line)
            f.write('\n')
            

#dialect_eval_files('../machamp/logs/en_xlmr/predictions', 'predictions/en_xlmr.conll.dialect.eval')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('please provide paths to prediction dir and output file')
    pred_dir = sys.argv[1]
    out_file1 = sys.argv[2]
    out_file2 = sys.argv[3]
    eval_files(pred_dir, out_file1)
    dialect_eval_files(pred_dir, out_file2)
    