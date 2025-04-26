import os
import re
import random
from word_swapping import update_spelling

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

input_folder = 'data/aux_data/ndc-aligned/aligned'
train_file = 'data/aux_data/ndc_train.txt'
dev_file = 'data/aux_data/ndc_dev.txt'

interviewers = ['ms', 'jb', 'ifg', 'rvf', 'sb', 'lks', 'mn', 'sl', 'sr',
                'kb', 'kh', 'iii', 'eo', 'hna', 'ma', 'os', 'as', 'ov',
                'amr', 'ran', 'mi', 'lh', 'mj', 'ahl', 'ks', 'amj', 'cbo',
                'jbj', 'jk', 'bl', 'ta', 'pmk', 'aml', 'amg']

skip_tokens = ['#', '##', '*', '…', '...', '?',  #pauses, overlaps
               'ee', 'eh', 'ehe', 'em', 'heh', 'hm', 'm', 'm-m', 'mhm', 'mm',  #interjections
               'hæ']

def process_file(file_path):
    sentences = []
    current_sent = []
    include_sent = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('<u id='):
                speaker_match = re.search(r'speaker="([^"]+)"', line)
                if speaker_match:
                    speaker = speaker_match.group(1)
                    if speaker in interviewers:
                        include_sent = False
                    else:
                        include_sent = True
                
            if line == "</u>":
                if include_sent and current_sent:
                    if len(current_sent) > 5:  #Only include sentences with more than 5 words
                        sentences.append(" ".join(current_sent))
                current_sent = []
            
            if include_sent and line:
                word = line.split("\t")[0]  
                 
                if word not in skip_tokens and not word.endswith('-') and not word.startswith('<'):
                    updated_word = update_spelling(word)
                    current_sent.append(updated_word)

    return sentences

all_sentences = []
for filename in os.listdir(input_folder):
    if filename.endswith(".vrt"):
        file_path = os.path.join(input_folder, filename)
        sentences = process_file(file_path)
        all_sentences.extend(sentences)


# Split 80-20 train dev
random.shuffle(all_sentences)
split_index = int(len(all_sentences) * 0.8)
train_sentences = all_sentences[:split_index]
dev_sentences = all_sentences[split_index:]

with open(train_file, "w") as train_out:
    for sentence in train_sentences:
        train_out.write(sentence + "\n")

with open(dev_file, "w") as dev_out:
    for sentence in dev_sentences:
        dev_out.write(sentence + "\n")

