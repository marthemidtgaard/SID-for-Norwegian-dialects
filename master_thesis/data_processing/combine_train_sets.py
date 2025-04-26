import random

def parse_conll_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    instances = []
    current_instance = []
    
    for line in lines:
        if line.strip():
            current_instance.append(line)
        else:
            if current_instance:
                instances.append(current_instance)
                current_instance = []
    
    if current_instance:
        instances.append(current_instance)
    
    return instances


def combine_conll_files(files, output_file):
    merged_instances = []
    
    for file in files:
        merged_instances.extend(parse_conll_file(file))

    random.seed(42)
    random.shuffle(merged_instances)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for instance in merged_instances:
            for line in instance:
                f.write(line)
            f.write('\n')  


combine_conll_files(['data/nomusic_dev.conll', 'data/noise_data/massive/mas_swapped_noise_75.conll', 'data/en.conll'], 'data/combined/full_nomusic_en_mas1.conll')
