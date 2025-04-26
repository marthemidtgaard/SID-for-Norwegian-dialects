from collections import Counter, defaultdict
import matplotlib.pyplot as plt
#from matplotlib_venn import venn3, venn2
import pandas as pd
#import numpy as np
import textwrap



def get_slot_counts(conll_file):
    # Initialize a list to store slot types
    slots = []

    # Open and process the file line by line
    with open(conll_file, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith("#") :  # Ignore empty lines and comments
                token_line = line.strip().split("\t")
                #print(token_line)
                slots.append(token_line[3])

    # Count slot occurrences
    slot_counts = Counter(slots)

    return dict(sorted(slot_counts.items()))

#for slot, count in get_slot_counts('data/en.conll').items():
#    print(slot)


def display_extra_and_missing_slots_compared_to_en(conll_file):
    slots_en = list(get_slot_counts('data/en.conll').keys())
    slots = list(get_slot_counts(conll_file).keys())

    extra_slots = set(slots) - set(slots_en)
    missing_slots = set(slots_en) - set(slots)

    # Display results
    print("Extra")
    for i in sorted(list(extra_slots)):
        print(i)

    print()
    print("Missing")
    for i in sorted(list(missing_slots)):
        print(i)


def get_tokens_for_tag(conll_file_path, input_tag):
    tokens = set()

    with open(conll_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() and not line.startswith("#"):  # Skip empty lines and metadata

                # Split the line into columns
                columns = line.split('\t')
                #print(columns)
                token = columns[1].strip()
                tag = columns[3].strip()
                #print(token,tag)

                if tag == input_tag:
                    tokens.add(token)
            
    return tokens

#set1 = get_tokens_for_tag('data/new1_nb.conll', 'B-datetime')
#print(set1)

def find_mismatched_slots(conll1, conll2, slot_to_check):
    def parse_conll(file):
        data = {}
        with open(file, 'r') as file:
            current_id = None
            current_text = None
            slots = []

            for line in file:
                line = line.strip()
                if line.startswith('# id'):
                    current_id = line.split(' = ')[1]
                elif line.startswith('# text'):
                    current_text = line.split(' = ')[1]
                elif line.startswith('#'):
                    continue
                elif not line:
                    slot_counts = Counter(slots)
                    data[current_id] = {'text': current_text, 'slots': slot_counts}
                    current_id = None
                    current_text = None
                    slots = []
                else:
                    token_line = line.split('\t')
                    slots.append(token_line[3])
            
            # Add the last entry if any
            if current_id and current_text:
                slot_counts = Counter(slots)
                data[current_id] = {"text": current_text, "slots": slot_counts}
        return data

    mismatched_utterances = []

    data1 = parse_conll(conll1)
    data2 = parse_conll(conll2)
    for id_ in data1.keys() | data2.keys():
        slots1 = data1.get(id_, {}).get("slots", Counter())
        slots2 = data2.get(id_, {}).get("slots", Counter())

        count1 = slots1.get(slot_to_check, 0)
        count2 = slots2.get(slot_to_check, 0)

        if count1 != count2:
            mismatched_utterances.append(id_)
    
    print(len(mismatched_utterances))
    print(mismatched_utterances[:10])

#find_mismatched_slots('data/en.conll', 'data/nb.conll', 'B-datetime')


def get_intents(file_path):
    unique_intents = set()
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("# intent = "):
                unique_intents.add(line.split(" = ")[1].strip())
    
    return unique_intents

#for intent in get_intents('data/nomusic_test.conll'):
#    print(intent)


def get_number_of_intents(file_path):
    intent_counts = defaultdict(int)
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("# intent = "): 
                intent = line.split(" = ")[1].strip()
                intent_counts[intent] += 1
                
    return dict(intent_counts)

#print(get_number_of_intents('data/en.conll'))

def get_slots_per_intent(file_path):
    #d = dict()
    d = defaultdict(lambda: defaultdict(int))
    current_intent = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('# intent'):
            current_intent = line.strip().split(' = ')[1]
            #current_intent = line.strip().split(' = ')[1]
            #if current_intent not in d:
                #d[current_intent] = set()

        elif line.strip() == '' or line.startswith('#'):
            continue
        else:
            if current_intent:
                slot = line.strip().split('\t')[3]    
                if '-' in slot:
                    slot = slot.split('-')[1]
                    #d[current_intent].add(slot)
                    d[current_intent][slot] += 1  # Count occurrences
    
    #for i, j in d.items():
    #    print(i, j)

    for intent in sorted(d.keys()):  
        print(f"{intent}")
        for slot, count in d[intent].items():
            print(f"{slot}, {count}")
        print()

#get_slots_per_intent('data/en.conll')


def process_conll_file(file_path):
    intent_data = defaultdict(lambda: {"slots": Counter(), "values": defaultdict(list)})
    current_intent = None
    slot_value = []
    slot_type = None  # Initialize slot_type to track current slot type

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                # Extract intent metadata
                if "intent =" in line:
                    intent_match = re.search(r"intent = \s*(\S+)", line)
                    if intent_match:
                        current_intent = intent_match.group(1)
            elif line and current_intent:
                # Process token lines if they have at least 4 columns
                parts = line.split()
                if len(parts) >= 4:
                    word, tag = parts[1], parts[3]
                    if tag.startswith("B-"):
                        # Save the previous slot if any
                        if slot_value and slot_type:
                            intent_data[current_intent]["values"][slot_type].append(" ".join(slot_value))
                        
                        # Start a new slot
                        slot_type = tag[2:]  # Set new slot type
                        slot_value = [word]  # Start a new slot value
                        intent_data[current_intent]["slots"][slot_type] += 1
                    elif tag.startswith("I-") and slot_value:
                        slot_value.append(word)
                    elif tag == "O" and slot_value:
                        # Join and save the completed slot value
                        intent_data[current_intent]["values"][slot_type].append(" ".join(slot_value))
                        slot_value = []  # Reset slot_value for the next slot
                        slot_type = None

    # Ensure the last slot value is saved if file ends without an 'O' tag
    if slot_value and slot_type:
        intent_data[current_intent]["values"][slot_type].append(" ".join(slot_value))
    
    return intent_data


def plot_weather_slot_values_info(intent_data):

    def get_slot_values(intent_data, intent, slot):
        if intent in intent_data and slot in intent_data[intent]["values"]:
            values = intent_data[intent]["values"][slot]
            values = [value.lower() for value in values]
            values_count = Counter(values)
            return values_count
        
        else:
            print(f"No values found for intent '{intent}' and slot '{slot}'.")

    weather_attr_train = weather_attr_train = get_slot_values(intent_data, "weather/find", "weather/attribute" )
    cond_desc_train = get_slot_values(intent_data, "weather/find", "condition_description" )
    cond_temp_train = get_slot_values(intent_data, "weather/find", "condition_temperature" )

    set1, set2, set3 = set(weather_attr_train.keys()), set(cond_desc_train.keys()), set(cond_temp_train.keys())

    unique_1 = set1 - set2 - set3
    unique_2 = set2 - set1 - set3
    unique_3 = set3 - set1 - set2
    in12 = set1 & set2 - set3
    in13 = set1 & set3 - set2

    subsets = {
        '100': set(),  
        '010': unique_2,  
        '001': unique_3,  
        '110': in12,      
        '101': in13       
    }
    unique_1 = ' '

    subsets_labels = {key: '\n'.join(values) if values else '' for key, values in subsets.items()}

    #--PLOT 1--#
    plt.figure(figsize=(10, 8))
    venn = venn3(subsets=(len(unique_1), len(unique_2), len(in12), 
                        len(unique_3), len(in13), 0, 0),  # Zero for unused sections
                set_labels=('weather/attribute', 'condition_description', 'condition_temperature'))

    for subset_id, label in subsets_labels.items():
        label_element = venn.get_label_by_id(subset_id)
        if label_element:  # Only set text if the section exists
            label_element.set_text(label)

    # Customize colors for better readability
    venn.get_patch_by_id('100').set_color('#FFB07C')
    venn.get_patch_by_id('100').set_alpha(0.7)

    venn.get_patch_by_id('010').set_color('#96DFCE')
    venn.get_patch_by_id('010').set_alpha(0.7)

    venn.get_patch_by_id('110').set_color('#FFE88C')
    venn.get_patch_by_id('110').set_alpha(0.7)

    venn.get_patch_by_id('101').set_color('#A2C2FF')
    venn.get_patch_by_id('101').set_alpha(0.6)

    venn.get_patch_by_id('001').set_color('#FF9E9E')
    venn.get_patch_by_id('001').set_alpha(0.6)


    # Title and show plot
    plt.savefig("overlapping_weather_slot_values.png", dpi=300, bbox_inches='tight')
    plt.show()


    #--PLOT 2--#
    def freq_plot(overlap_set, freq_counter1, freq_counter2, slot1, slot2, name):
        bar_width = 0.35
        x = np.arange(len(overlap_set))  # Label locations
        plt.figure(figsize=(10, 6))

        bars1 = plt.bar(x - bar_width / 2, freq_counter1, bar_width, label=slot1, color='#FFB07C', alpha=1)
        bars2 = plt.bar(x + bar_width / 2, freq_counter2, bar_width, label=slot2, color='#FFE88C', alpha=1)

        # Add text on top of each bar
        for bar in bars1:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

        # Set y-axis to logarithmic scale to handle tall bars better
        plt.yscale('log')

        # Add labels, title, and custom x-axis tick labels
        #plt.xlabel('Overlapping Strings')
        #plt.ylabel('Frequency')
        plt.xticks(x, overlap_set, rotation=45)
        plt.legend()

        # Display the plot
        plt.tight_layout()
        plt.savefig(f"barplot_{name}.png", dpi=300, bbox_inches='tight')
        plt.show()

    
    freq_counter1 = [weather_attr_train[slot] for slot in in12]
    freq_counter2 = [cond_desc_train[slot] for slot in in12]
    freq_plot(in12, freq_counter1, freq_counter2, 'weather/attribute', 'condition_description', 'weater_attr_condistion_descr')

    freq_counter1 = [weather_attr_train[slot] for slot in in13]
    freq_counter3 = [cond_temp_train[slot] for slot in in13]
    freq_plot(in13, freq_counter1, freq_counter3, 'weather/attribute', 'condition_temperature', 'weater_attr_condistion_temp')

#intent_data = process_conll_file("data/en.conll")
#plot_weather_slot_values_info(intent_data)

def plot_overlap_facebook_snips():
    dataset1_slots = {
        "recurring_datetime", "reference", "datetime", "negation", "alarm_modifier",
        "timer_attributes", "reminder_todo", "temperature_unit", "location", "weather_attribute"
    }
    dataset2_slots = {
        "reference", "datetime", "location", "entity_name", "playlist", "artist", 
        "music_item", "cuisine", "served_dish", "restaurant_name", "party_size_number",
        "party_size_description", "facility", "restaurant_type", "sort", "service",
        "track", "album", "genre", "best_rating", "rating_unit", "object_type",
        "object_name", "rating_value", "object_select", "series_type",
        "movie_name", "location_type", "movie_type"
    }


    #Generate the Venn diagram
    plt.figure(figsize=(10, 8))
    venn = venn2(subsets=(len(dataset1_slots) * 1.2, len(dataset2_slots), len(dataset1_slots & dataset2_slots) * 2), set_labels=('Facebook', 'SNIPS'))

    #Helper functions to wrap text
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width))

    def wrap_text_per_line(text_set):
        return ",\n".join(text_set)


    # Format and place text in each section
    venn.get_label_by_id('10').set_text(wrap_text(", ".join(dataset1_slots - dataset2_slots), 25))
    venn.get_label_by_id('01').set_text(wrap_text(", ".join(dataset2_slots - dataset1_slots), 25))
    venn.get_label_by_id('11').set_text(wrap_text_per_line(dataset1_slots & dataset2_slots))


    #Colors
    venn.get_patch_by_id('10').set_color('#FFB07C')
    venn.get_patch_by_id('10').set_alpha(0.7)
    venn.get_patch_by_id('01').set_color('#96DFCE')
    venn.get_patch_by_id('01').set_alpha(0.7)
    venn.get_patch_by_id('11').set_color('#FFE88C')
    venn.get_patch_by_id('11').set_alpha(0.7)


    plt.savefig("overlapping_slots_facebook_snips.png", dpi=300, bbox_inches='tight')
    plt.show()