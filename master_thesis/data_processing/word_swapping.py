import os
from collections import defaultdict
import random
import json
import re

def is_vowel(char):
    return char.lower() in "aeiouyåæø"


def is_consonant(char):
    return not is_vowel(char)


def remove_short_vowel_consonant_cluster(token):
    """If a double consonant is followed by a third conosonant, remove one of the double consonants. C1C1C2 -> C1C2"""    

    prev_char = token[0]
    double_cons = False
    offset = 0
    for i, cur_char in enumerate(token):
        if i == 0:
            continue
        if is_consonant(cur_char):
            if cur_char == prev_char:
                double_cons = True
            elif double_cons:
                if not (prev_char in "sk" and cur_char == "j"):
                    token = token[:i - 1 - offset] + token[i - offset:]
                    offset += 1
                double_cons = False
            else:
                double_cons = False
        else:
            double_cons = False
        prev_char = cur_char
    return token

def replace_å_double_consonant(token):
    """
    If the letter 'å' is immediately followed by a double consonant,
    change 'å' to 'o'. For example: "ståppe" -> "stoppe".
    """

    # Pattern matches 'å' followed by a double of a consonant
    pattern = re.compile(r"å([^aeiouyæøå])\1")  #\1 = match exactly the same character that was captured by (...)

    #Replace 'å' with 'o' while keeping the double consonant intact.
    return pattern.sub(lambda m: "o" + m.group(1)*2, token)


def update_spelling(token):
    """Make the transcription more similar to actual writing"""

    # Remove syllabic consonants, syllable boundaries
    token = token.replace("'", "")

    #Change Å to O if followed by a double consonant
    token = replace_å_double_consonant(token)

    #Simplify specific consonant sequences that indicate short vowels. Not typically seen in standard writing
    token = token.replace("ssjt", "rst")
    token = token.replace("ssjk", "rsk")
    token = token.replace("nngk", "nk")
    token = token.replace("ngk", "nk")

    #Remove consonant from a double consonant (from short vowel marking) if it is followed by another consonant.
    if len(token) > 3 and token not in ["ikkje", "issje"]:
        # NB this catches some false positives
        token = remove_short_vowel_consonant_cluster(token)

    # Change the retroflex flap to normal "l"
    token = token.replace("L", "l")

    token = token.replace("_", " ")

    return token



def create_bm_to_dialects_dict(folder_path):
    bm_to_dialects = defaultdict(lambda: {"north norwegian": set(), "west norwegian": set(), "trøndersk": set()})
                                          
    skip_tokens = ['#', '##', '*', '…', '...', '?',  #pauses, overlaps
                  'ee', 'eh', 'ehe', 'em', 'heh', 'hm', 'm', 'm-m', 'mhm', 'mm'  #interjections
                  ]
    
    interviewers = ['ms', 'jb', 'ifg', 'rvf', 'sb', 'lks', 'mn', 'sl', 'sr',
                    'kb', 'kh', 'iii', 'eo', 'hna', 'ma', 'os', 'as', 'ov',
                    'amr', 'ran', 'mi', 'lh', 'mj', 'ahl', 'ks', 'amj', 'cbo',
                    'jbj', 'jk', 'bl', 'ta', 'pmk', 'aml', 'amg']
    
    west = ['aure', 'bud', 'heroey', 'rauma', 'stranda', 'surnadal', 'todalen', 'volda', 
            'evje', 'landvik', 'valle', 'vegaardshei',
            'kristiansand', 'lyngdal', 'sirdal', 'vennesla', 'aaseral',
            'hyllestad', 'joelster', 'kalvaag', 'luster', 'stryn', 
            'gjesdal', 'hjelmeland', 'karmoey', 'sokndal', 'stavanger', 'suldal', 'time',
            'bergen', 'boemlo', 'eidfjord', 'fusa', 'kvinnherad', 'lindaas', 'voss',]
                  
    trøndersk = ['bjugn', 'gauldal', 'oppdal', 'roeros', 'selbu', 'skaugdalen', 'stokkoeya', 'trondheim',
                'inderoy', 'lierne', 'meraaker', 'namdalen']

    north = ['botnhamn', 'karlsoey', 'kirkesdalen', 'maalselv', 'kvaefjord', 'kaafjord', 
             'lavangen', 'medby', 'mefjordvær', 'stonglandseide', 'tromsoe']

    speaker_pattern = re.compile(r'speaker="([^"]+)"')
    process_tokens = True

    for filename in os.listdir(folder_path):
        if filename.endswith('.vrt'):
            dialect_tag = None
            file_prefix = filename.split('_')[0].lower()

            if file_prefix in west:
                dialect_tag = "west norwegian"
            elif file_prefix in trøndersk:
                dialect_tag = "trøndersk"
            elif file_prefix in north:
                dialect_tag = "north norwegian"
            
            if not dialect_tag:
                continue

            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    
                    line = line.strip()
                    if line.startswith('<u '):
                        match = speaker_pattern.search(line)
                        if match:
                            speaker = match.group(1)
                            if speaker in interviewers:
                                process_tokens = False
                            else:
                                process_tokens = True
                    
                    if line.startswith('</'):
                        process_tokens = True
                        continue
                
                    if process_tokens:  #process_tokens is only true if speaker is not an interviewer
                        parts = line.split('\t')
                        if len(parts) == 2:
                            dialect_word, bm_word = parts
                            dialect_word = dialect_word.lower()
                            bm_word = bm_word.lower()
                            if dialect_word != bm_word:
                    
                                dialect_word = update_spelling(dialect_word)
                                if dialect_word not in skip_tokens and bm_word not in skip_tokens:
                                    if dialect_word != bm_word:
                                        if not bm_word.endswith('-') and not dialect_word.endswith('-'):   #indicates interrupted words
                                            if len(bm_word) == 1 or len(dialect_word) > 1:
                                                bm_to_dialects[bm_word][dialect_tag].add(dialect_word)
        
    bm_to_dialects = {key: {d: list(v) for d, v in value.items() if v} for key, value in bm_to_dialects.items()}

    with open('bm_to_dialects.json', 'w', encoding='utf-8') as json_file:
        json.dump(bm_to_dialects, json_file, ensure_ascii=False, indent=4)

    return bm_to_dialects



def swap_words(input_file, output_file, bm_to_dialects, num_ids, seed=42):
    random.seed(seed)  # Set the random seed for reproducibility
    swap_count = 0

    # 1) Generate list of ids
    all_ids = [str(i) for i in range(1, num_ids + 1)]  # Assuming IDs are sequential numbers as strings
    random.shuffle(all_ids)

    # 2) Distribute ids into dialect groups
    num_north = int(0.2 * num_ids)
    num_trønder = int(0.3 * num_ids)
    num_west = num_ids - (num_north + num_trønder)  # Assign remaining to west (50%)

    north_ids = all_ids[:num_north]
    trøndersk_ids = all_ids[num_north:num_north + num_trønder]
    west_ids = all_ids[num_north + num_trønder:]

    dialect_tag = None  # Keep track of the dialect for each sentence

    # 3) Swap words
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('# id'):
                id  = line.strip().split(' = ')[1]
                if id in north_ids:
                    dialect_tag = 'north norwegian'
                elif id in west_ids:
                    dialect_tag = 'west norwegian'
                elif id in trøndersk_ids:
                    dialect_tag = 'trøndersk'

                outfile.write(line)
                outfile.write(f'# comment = Swapped to {dialect_tag}\n')


            elif line.startswith('#'):
                outfile.write(line)
            elif line.strip() == '':
                outfile.write(line)
                dialect_tag = None
            
            else:
                parts = line.strip().split('\t')
                token = parts[1]
                if token in bm_to_dialects and dialect_tag in bm_to_dialects[token]:
                    new_token = random.choice(bm_to_dialects[token][dialect_tag])
                    parts[1] = new_token
                    swap_count += 1
                    outfile.write('\t'.join(parts) + '\n')
                    #print(token, new_token)
                else:
                    outfile.write(line)
    
    print(f"\nTotal words swapped: {swap_count}")


bm_to_dialects = create_bm_to_dialects_dict('data/aux_data/ndc-aligned/aligned')
#swap_words('data/nb_rt1.conll', 'data/nb_rt1_swapped.conll', bm_to_dialects, 2021)
swap_words('data/mas.conll', 'data/mas_swapped.conll', bm_to_dialects, 2021)


