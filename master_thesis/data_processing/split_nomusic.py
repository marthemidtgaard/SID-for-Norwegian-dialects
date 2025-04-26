import random
import re

def read_conll_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = file.read().strip()
    instances = data.split("\n\n")  
    return instances

def group_instances_by_id(instances):
    grouped = {}
    pattern = re.compile(r"# id = (\d+)/")
    
    for instance in instances:
        match = pattern.search(instance)
        if match:
            group_id = match.group(1)
            if group_id not in grouped:
                grouped[group_id] = []
            grouped[group_id].append(instance)
    
    return list(grouped.values())

def split_data(grouped_instances, train_ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(grouped_instances) 
    
    split_idx = int(len(grouped_instances) * train_ratio)
    train_groups = grouped_instances[:split_idx]
    test_groups = grouped_instances[split_idx:]
    
    train_data = [instance for group in train_groups for instance in group]
    test_data = [instance for group in test_groups for instance in group]

    return train_data, test_data


def write_conll_file(filepath, instances):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\n\n".join(instances))


instances = read_conll_file('data/nomusic_dev.conll')
grouped_instances = group_instances_by_id(instances)
train_data, test_data = split_data(grouped_instances)
    
write_conll_file('data/nomusic_train.conll', train_data)
write_conll_file('data/nomusic_new_dev.conll', test_data)

print(f"Data split complete. Train: {len(train_data)} instances, Test: {len(test_data)} instances.")

