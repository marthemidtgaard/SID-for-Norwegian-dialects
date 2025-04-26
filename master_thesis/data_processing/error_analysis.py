import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict, Counter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def create_dataframe(gold_file, pred_file=None):
    def parse_gold_conll(file_path):
        sentences = []
        intents = []
        slots = []
        ids = []
        
        with open(file_path, 'r') as f:
            current_tokens = []
            current_slots = []
            current_intent = None
            current_id = None

            for i, line in enumerate(f):
                if line.startswith("# id"):
                    current_id = line.split(" = ")[1].strip()  # More robust way to get ID
                elif line.startswith("# intent"):
                    current_intent = line.strip().split(" ")[-1]  #normally " = "
                elif line.startswith("# text") or line.startswith("# dialect") or line.startswith("# slots"):
                    continue
                elif line.strip() == "":
                    # End of sentence, store the current sentence data
                    if current_tokens and current_id is not None:
                        sentences.append(current_tokens)
                        slots.append(current_slots)
                        intents.append(current_intent)
                        ids.append(current_id)
                        current_tokens = []
                        current_slots = []
                else:
                    # Columns: token, intent (for this token), slot label
                    columns = line.strip().split("\t")
                    _, token, intent, slot = columns
                    current_tokens.append(token)
                    current_slots.append(slot)
                    
            # Catch the last sentence if file doesn't end with a blank line
            if current_tokens:
                sentences.append(current_tokens)
                slots.append(current_slots)
                intents.append(current_intent)
                ids.append(current_id)

        return sentences, intents, slots, ids

    def parse_pred_conll(file_path):
        sentences = []
        intents = []
        slots = []
        
        with open(file_path, 'r') as f:
            current_tokens = []
            current_slots = []
            current_intent = None

            for i, line in enumerate(f):
                if line.startswith("# intent"):
                    current_intent = line.strip().split(" ")[-1]  #normally " = "
                elif line.startswith("# text") or line.startswith("# id") or line.startswith("# dialect") or line.startswith("# slots"):
                    continue
                elif line.strip() == "":
                    # End of sentence, store the current sentence data
                    if current_tokens:
                        sentences.append(current_tokens)
                        slots.append(current_slots)
                        intents.append(current_intent)
                        current_tokens = []
                        current_slots = []

                else:
                    # Columns: token, intent (for this token), slot label
                    columns = line.strip().split("\t")
                    _, token, intent, slot = columns
                    current_tokens.append(token)
                    current_slots.append(slot)
                    
            # Catch the last sentence if file doesn't end with a blank line
            if current_tokens:
                sentences.append(current_tokens)
                slots.append(current_slots)
                intents.append(current_intent)

        return sentences, intents, slots

    # Parse the gold and predicted files
    gold_sentences, gold_intents, gold_slots, gold_ids = parse_gold_conll(gold_file)
    if pred_file:
        pred_sentences, pred_intents, pred_slots = parse_pred_conll(pred_file)

    # Check for alignment
    if pred_file:
        if len(gold_sentences) != len(pred_sentences):
            #print(len(gold_sentences), len(pred_sentences))
            raise ValueError("Mismatch in number of sentences between gold and predicted files.")

    # Create a DataFrame
    if pred_file:
        data = {
            'sentence': gold_sentences,  # Sentence tokens (should be same in both)
            'id': gold_ids,  # Should be same in both
            'gold_intent': gold_intents,
            'pred_intent': pred_intents,
            'gold_slots': gold_slots,
            'pred_slots': pred_slots
        }
    else:
        data = {
        'sentence': gold_sentences,  # Sentence tokens (should be same in both)
        'id': gold_ids,  # Sentence tokens (should be same in both)
        'gold_intent': gold_intents,
        'gold_slots': gold_slots,
    }

    df = pd.DataFrame(data)
    return df


def get_misclassified_intents(df):
    intent_mispredictions = df[df['gold_intent'] != df['pred_intent']]
    misclassified_data = defaultdict(lambda: {"count": 0, "ids": []})

    for _, row in intent_mispredictions.iterrows():
        pair = (row['pred_intent'], row['gold_intent'])
        misclassified_data[pair]["count"] += 1
        misclassified_data[pair]["ids"].append(row["id"])

    # Sort by frequency (highest first)
    sorted_misclassified_data = dict(sorted(misclassified_data.items(), key=lambda x: x[1]["count"], reverse=True))

    return sorted_misclassified_data


def get_misclassified_slots(df):
    slot_mispredictions = df[df.apply(lambda row: row['gold_slots'] != row['pred_slots'], axis=1)]
    misclassified_data = defaultdict(lambda: {"count": 0, "ids": []})

    for _, row in slot_mispredictions.iterrows():
        pred_slots = row['pred_slots']
        gold_slots = row['gold_slots']
        for pred_slot, gold_slot in zip(pred_slots, gold_slots):
            if pred_slot != gold_slot:
                pair = (pred_slot, gold_slot)
                misclassified_data[pair]["count"] += 1
                misclassified_data[pair]["ids"].append(row["id"])

    # Sort by frequency (highest first)
    sorted_misclassified_data = dict(sorted(misclassified_data.items(), key=lambda x: x[1]["count"], reverse=True))

    return sorted_misclassified_data


def compare_misclassified_across_files(pred_dfs, csv_file_path, intents=False, slots=False):
    #pred_dfs (dict): Dictionary where keys are filenames and values are DataFrames with predicted intents.
    all_misclassifications = defaultdict(lambda: defaultdict(int))

    for file_name, pred_df in pred_dfs.items():
        if intents:
            misclassified_data = get_misclassified_intents(pred_df)
        elif slots:
            misclassified_data = get_misclassified_slots(pred_df)

        for (pred, gold), data in misclassified_data.items():
            all_misclassifications[(pred, gold)][file_name] = data["count"]

    # Convert to DataFrame
    comparison_df = pd.DataFrame.from_dict(all_misclassifications, orient="index").fillna(0)

    # Sort by total misclassification frequency
    comparison_df["Total"] = comparison_df.sum(axis=1)
    comparison_df = comparison_df.sort_values(by="Total", ascending=False)

    comparison_df.to_csv(csv_file_path, index=True)
    return comparison_df


def find_id_overlaps_for_errors(pred_files, intents=False, slots=False):
    misclassification_ids = defaultdict(lambda: defaultdict(set))

    for file_name, pred_df in pred_files.items():
        if intents:
            # Get misclassified intents with their IDs
            misclassified_data = get_misclassified_intents(pred_df)
        elif slots:
            misclassified_data = get_misclassified_slots(pred_df)

        for error, data in misclassified_data.items():
            misclassification_ids[error][file_name] = set(data["ids"])

    # Find overlaps for each error pair (at least 2 overlapping)
    """overlaps = {}
    for error, file_ids in misclassification_ids.items():
        common_ids = set.intersection(*file_ids.values()) if len(file_ids) > 1 else set()
        overlaps[error] = {
            "overlapping_ids": common_ids,
            "file_specific_ids": file_ids
        }

    # Convert results to DataFrame for easy viewing
    error_overlaps_df = pd.DataFrame([
        {"Error Pair": error, "Overlapping IDs": data["overlapping_ids"], "File-Specific IDs": data["file_specific_ids"]}
        for error, data in overlaps.items()
    ])"""
    overlaps = {}
    for error, file_ids in misclassification_ids.items():
        all_id_sets = list(file_ids.values())

        # Overlapping IDs in at least `min_overlap` files
        common_ids = set.intersection(*all_id_sets) if len(all_id_sets) > 1 else set()

        # Full overlap: IDs that appear in ALL files
        full_overlap_ids = set.intersection(*all_id_sets) if len(all_id_sets) == len(pred_files) else set()

        overlaps[error] = {
            "overlapping_ids": common_ids,
            "file_specific_ids": file_ids,
            "full_overlap_ids": full_overlap_ids
        }

    # Convert results to DataFrame for easy viewing
    error_overlaps_df = pd.DataFrame([
        {
            "Error Pair": error,
            "Overlapping IDs": data["overlapping_ids"],
            "File-Specific IDs": data["file_specific_ids"],
            "Full Overlap IDs": data["full_overlap_ids"]
        }
        for error, data in overlaps.items()
    ])
    return error_overlaps_df


def get_percentage_misclassified_intents(pred_dfs):
    for model_name, df in pred_dfs.items():
        #print(model_name)
        
        intent_counts = df['gold_intent'].value_counts().sort_index()
        #print("Intent counts:")
        #print(intent_counts)

        #print("\nMisprediction rates per intent:")
        for intent, count in intent_counts.items():
            intent_df = df[df['gold_intent'] == intent]
            num_wrong = (intent_df['gold_intent'] != intent_df['pred_intent']).sum()
            #print(num_wrong)
            #percent_wrong = (num_wrong / count) * 100
            #print(f"  {intent}: {percent_wrong:.2f}% mispredicted ({num_wrong}/{count})")


def print_ids_for_errors(pred_files, intents=False, slots=False):
    # {(gold_intent, id): {pred_intent: set(models)}}
    misclassification_map = defaultdict(lambda: defaultdict(set))
    id_to_sentence = {}

    for model_name, pred_df in pred_files.items():
        if intents:
            misclassified_data = get_misclassified_intents(pred_df)
        elif slots:
            misclassified_data = get_misclassified_slots(pred_df)
        else:
            raise ValueError("You must set either intents=True or slots=True")

        for (pred_intent, gold_intent), data in misclassified_data.items():
            for id_val in data["ids"]:
                key = (gold_intent, id_val)
                misclassification_map[key][pred_intent].add(model_name)

        # Store id-to-sentence mapping (for all ids in the DataFrame)
        for _, row in pred_df.iterrows():
            id_to_sentence[row["id"]] = row["sentence"]

    # Organize mispredictions
    organized = defaultdict(list)
    for (gold_intent, id_val), pred_dict in misclassification_map.items():
        for pred_intent, model_set in pred_dict.items():
            organized[gold_intent].append((id_val, pred_intent, sorted(model_set)))

    # Print mispredictions with sentence
    for gold_intent, mispredictions in organized.items():
        print(f"\nGold intent: {gold_intent}")
        id_to_preds = defaultdict(list)
        for id_val, pred_intent, models in mispredictions:
            id_to_preds[id_val].append((pred_intent, models))

        for id_val, preds in id_to_preds.items():
            sentence = id_to_sentence.get(id_val, "[sentence not found]")
            sentence = ' '.join(sentence)
            print(f"  ID: {id_val}")
            print(f"     Sentence: {sentence}")
            if len(preds) == 1:
                pred_intent, models = preds[0]
                print(f"     → Predicted: {pred_intent} | Model(s): {', '.join(models)}")
            else:
                print(f"     → Multiple conflicting predictions:")
                for pred_intent, models in preds:
                    print(f"        - {pred_intent} | {', '.join(models)}")

        print()


def get_percentage_misclassified(pred_dfs, intents=False, slots=False):
    for model_name, df in pred_dfs.items():
        print(model_name)
        
        if intents:
            counts = df['gold_intent'].value_counts().sort_index()
            print(counts)

            for intent, count in counts.items():
                if intents:
                    intent_df = df[df['gold_intent'] == intent]
                    num_wrong = (intent_df['gold_intent'] != intent_df['pred_intent']).sum()
                    print(num_wrong)


        elif slots:
            slot_error_counts = {}
            total_label_counts = {}

            for gold_slots, pred_slots in zip(df['gold_slots'], df['pred_slots']):
                for g, p in zip(gold_slots, pred_slots):
                    total_label_counts[g] = total_label_counts.get(g, 0) + 1
                    if g != p:
                        slot_error_counts[g] = slot_error_counts.get(g, 0) + 1

            for label in sorted(total_label_counts):
                total = total_label_counts[label]
                wrong = slot_error_counts.get(label, 0)
                print(f"{label},{total},{wrong}")
        
        print()


def plot_intent_heatmap_pair(df1, name1, df2, name2, task=None, slot_prefix=None):
    #fig = plt.figure(figsize=(12, 8))
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)

    #gs = GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.6)  # for intents
    gs = GridSpec(2, 2, figure=fig, wspace=0.01, hspace=0.01)  # for slots


    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    def create_cmap():
        cmap = LinearSegmentedColormap.from_list("custom_gradient", ["#FFF7E3", "#FFB07C", "#FF9050", "#FF7040", "#FF5733"])
        return cmap

    def compute_cm_data(df, task=None, slot_prefix=None):
        if task == 'intents':
            cm = pd.crosstab(df['gold_intent'], df['pred_intent'])
        elif task == 'slots':
            gold_all = []
            pred_all = []

            for gold, pred in zip(df['gold_slots'], df['pred_slots']):
                for g, p in zip(gold, pred):
                    if slot_prefix is None or g.startswith(slot_prefix):
                        gold_all.append(g)
                        pred_all.append(p)

            cm = pd.crosstab(pd.Series(gold_all, name="GOLD"), pd.Series(pred_all, name="PRED"))

                
        cm_norm = cm.div(cm.sum(axis=1), axis=0) * 100
        return cm, cm_norm



    def get_off_diag_nonzero_values(cm):
        values = []
        for i in cm.index:
            for j in cm.columns:
                if i != j and cm.loc[i, j] != 0:
                    values.append(cm.loc[i, j])
        return values

    # Compute confusion matrices
    cm1, cm1_norm = compute_cm_data(df1, task=task, slot_prefix=slot_prefix)
    cm2, cm2_norm = compute_cm_data(df2, task=task, slot_prefix=slot_prefix)

    # Compute vmax values based on off-diagonal non-zero cells
    counts_all = get_off_diag_nonzero_values(cm1) + get_off_diag_nonzero_values(cm2)
    norms_all = get_off_diag_nonzero_values(cm1_norm) + get_off_diag_nonzero_values(cm2_norm)

    max_value_count = max(counts_all) if counts_all else 1
    max_value_norm = max(norms_all) if norms_all else 1e-6


    def plot_single_heatmap(data, ax, fmt, vmin, vmax):
        # Create mask for diagonal and zeros
        diag_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
        common_labels = data.index.intersection(data.columns)
        for label in common_labels:
            diag_mask.loc[label, label] = True
        zero_mask = data == 0
        combined_mask = diag_mask | zero_mask
        combined_mask = combined_mask.reindex_like(data)

        if fmt == '.1f':
            annot_data = data.copy().applymap(lambda x: f"{x:.1f}%" if x != 0 else "")
            fmt = ''
        else:
            annot_data = True
    
        # Plot
        sb.heatmap(data,
                   ax=ax,
                   annot=annot_data,
                   fmt=fmt,
                   cmap=create_cmap(),
                   mask=combined_mask,
                   cbar=False,
                   linewidths=0.5,
                   linecolor='gray',
                   annot_kws={"size": 6.5, "color": "black"},
                   vmin=vmin,
                   vmax=vmax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    # Plotting each quadrant
    plot_single_heatmap(cm1_norm, ax00, fmt='.1f', vmin=0, vmax=max_value_norm)
    ax00.set_title(f'{name1} (Normalized)', fontsize=14)

    plot_single_heatmap(cm1, ax01, fmt='g', vmin=0, vmax=max_value_count)
    ax01.set_title(f'{name1} (Counts)', fontsize=14)

    plot_single_heatmap(cm2_norm, ax10, fmt='.1f', vmin=0, vmax=max_value_norm)
    ax10.set_title(f'{name2} (Normalized)', fontsize=14)

    plot_single_heatmap(cm2, ax11, fmt='g', vmin=0, vmax=max_value_count)
    ax11.set_title(f'{name2} (Counts)', fontsize=14)

    fig.text(0.5, 0.07, 'PRED', ha='center', fontsize=14)
    fig.text(0.07, 0.5, 'GOLD', va='center', rotation='vertical', fontsize=14)


    #plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.savefig("intent_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


def barplot_slots(df, data, show_percentage=True, min_count=10):
    def slot_sort_key(label):
        group = 0 if label.startswith("B-") else 1 if label.startswith("I-") else 2 if label == "O" else 3
        return (group, -misclassification_data[label])  # sort within group by descending proportion

    #Explode to token level and only keep misclassifications
    df_exploded = df.explode(['sentence', 'gold_slots', 'pred_slots'])
    df_exploded = df_exploded[df_exploded['gold_slots'] != df_exploded['pred_slots']]

    #Filter out predicted slots not in gold slots
    unique_gold_slots = df_exploded['gold_slots'].unique()
    df_exploded = df_exploded[df_exploded['pred_slots'].isin(unique_gold_slots)]

    total_gold_counts = df.explode(['gold_slots'])['gold_slots'].value_counts()
    misclassified_counts = df_exploded['gold_slots'].value_counts()

    if show_percentage:
        misclassification_data = (misclassified_counts / total_gold_counts).fillna(0)
        misclassification_data = misclassification_data[misclassification_data >= 0.10]
    else:
        misclassification_data = misclassified_counts[misclassified_counts >= min_count]


    sorted_labels = sorted(misclassification_data.index, key=slot_sort_key)
    misclassification_data = misclassification_data.reindex(sorted_labels)

    # Create DataFrame for seaborn
    bar_data = pd.DataFrame({
        'label': misclassification_data.index,
        'value': misclassification_data.values
    })

    
    plt.figure(figsize=(12, 6))    

    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", ["#FFF7E3", "#FFB07C", "#FF9050", "#FF7040", "#FF5733"])
    norm = plt.Normalize(bar_data['value'].min(), bar_data['value'].max())
    mapped_colors = dict(zip(bar_data['label'], [custom_cmap(norm(v)) for v in bar_data['value']]))

    # Plot
    plt.figure(figsize=(14, 6))
    bar_plot = sb.barplot(
        data=bar_data,
        x='label',
        y='value',
        hue='label',
        palette=mapped_colors,
        legend=False
    )

    if show_percentage:
        plt.ylabel('')
        plt.xlabel('')
    else:
        plt.ylabel('')
        plt.xlabel('', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Set y-axis limit
    if show_percentage:
        plt.ylim(0, 1)
    else:
        plt.yscale('log')
        plt.ylim(0, bar_data['value'].max() * 1.1)


    # Add proportion labels on top of the bars
    for bar in bar_plot.patches:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        label = f'{height:.2f}' if show_percentage else f'{int(height)}'

        if show_percentage:
            offset = height * 0.02
            offset = max(offset, 0.01)
        else:
            offset = 20
        
        # Label placement
        if height >= (0.95 if show_percentage else bar_data['value'].max() * 0.95):
            # Put label inside the bar for tall bars
            plt.text(x, height - offset, label, ha='center', va='top', fontsize=12, color='black')
        else:
            # Normal placement above the bar
            plt.text(x, height, label, ha='center', va='bottom', fontsize=12, color='black')
    
    plt.tight_layout()
    if show_percentage:
        plt.savefig(f"slots_barplot_percentage_{data}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"slots_barplot_counts_{data}.png", dpi=300, bbox_inches='tight')
    plt.show()


def heatmap_slots(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    
    # Pivot table for heatmap
    heatmap_data = df.pivot_table(index="gold", columns="predicted", values="mispredicted", fill_value=0)
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", ["#FFF7E3", "#FFB07C", "#FF9050", "#FF7040", "#FF5733"])

    # Generate heatmap with custom color palette and zero masking
    mask = heatmap_data == 0

    plt.figure(figsize=(14, 10))
    sb.heatmap(
        heatmap_data,
        annot=True,
        fmt='g',
        cmap=custom_cmap,
        cbar=False,
        mask=mask,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"color": "black"},
    )

    plt.xlabel("PRED", fontsize=14)
    plt.ylabel("GOLD", fontsize=14, labelpad=2)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("slots_heatmap.png", dpi=300, bbox_inches='tight')
    #plt.show()
    

def print_slot_mismatches(df, gold_slot, pred_slot):
    same_intent = 0
    diff_intent = 0
    for idx, row in df.iterrows():
        gold_slots = row['gold_slots']
        pred_slots = row['pred_slots']
        sentence = row['sentence']
        gold_intent = row['gold_intent']
        pred_intent = row['pred_intent']

        
        for token, g_slot, p_slot in zip(sentence, gold_slots, pred_slots):
            if g_slot == gold_slot and p_slot == pred_slot:
                if gold_intent == pred_intent:
                    same_intent += 1
                else:
                    diff_intent += 1

                print(f"\nSentence: {' '.join(sentence)}")
                print(f"Gold: {gold_intent}, Pred: {pred_intent}")
                print(f"Gold Slots: {' '.join(gold_slots)}")
                print(f"Pred Slots: {' '.join(pred_slots)}")
                break  # Only print once per sentence

    #print(f"Same Intent: {same_intent}, Different Intent: {diff_intent}")


pred_dfs = dict()
pred_files = ['predictions/test_set/weather/en_nb_rt1_noise_swap_xlmr/2025.03.19_01.58.32.conll', 
                'predictions/test_set/weather/nomusic_en_mas1_xlmr/2025.03.19_21.31.08.conll', 
                ]

for file in pred_files:
    filename = file.split('/')[3]
    pred_df = create_dataframe("data/nomusic_test_weather.conll", file)
    pred_dfs[filename] = pred_df


#----------INTENTS ANALYSIS----------#
#compare_misclassified_across_files(pred_dfs, 'intent_misclassifications.csv', intents=True)
#get_percentage_misclassified(pred_dfs, intents=True)

print_ids_for_errors(pred_dfs, intents=True)
#plot_intent_heatmap_pair(pred_dfs['en_nb_rt1_noise_swap_xlmr'], 'en_nb_rt1_noise_swap_xlmr', pred_dfs['en_mas1_nomusic_xlmr'], 'en_mas1_nomusic_xlmr', task='intents')



#----------SLOTS ANALYSIS----------#
#compare_misclassified_across_files(pred_dfs, 'slot_misclassifications.csv', slots=True)
#get_percentage_misclassified(pred_dfs, slots=True)

#barplot_slots(pred_dfs['en_mas1_nomusic_xlmr'], 'en_mas1_nomusic_xlmr', show_percentage=False)
#barplot_slots(pred_dfs['en_mas1_nomusic_xlmr'], 'en_mas1_nomusic_xlmr', show_percentage=True)

#print_slot_mismatches(pred_dfs['en_mas1_nomusic_xlmr'], 'B-datetime', 'O')
