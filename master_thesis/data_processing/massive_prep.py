import pandas as pd
import json

#--1) Find the English utterances from MAS:de-ba
def conll_to_df(path: str):
    # Initialize lists to store data
    data = {
        'id': [],
        'text_EN': [],
        'intent': [],
        'intent_massive': [],
    }

    # Open the MAS:de-ba .conll file 
    with open(path, 'r') as file:
        lines = file.readlines()

    current_utt = {'id': None, 'text_EN': None, 'intent': None, 'intent_massive': None}

    for line in lines:
        if line.startswith('# id:'):
            current_utt['id'] = line.strip().split(': ')[1]

        elif line.startswith('# text-en:'):
            current_utt['text_EN'] = line.strip().split(': ')[1]

        elif line.startswith('# intent:'):
            current_utt['intent'] = line.strip().split(': ')[1]
        
        elif line.startswith('# intent-massive:'):
            current_utt['intent_massive'] = line.strip().split(': ')[1]

        # Empty line marking a new utterance on the next line
        elif line.strip() == '':

            data['id'].append(current_utt['id'])
            data['text_EN'].append(current_utt['text_EN'])
            data['intent'].append(current_utt['intent'])
            data['intent_massive'].append(current_utt['intent_massive'])
    
    return pd.DataFrame(data)

valid_utts = conll_to_df('data/massive/de-ba.MAS.valid.conll')
test_utts = conll_to_df('data/massive/e-ba.MAS.test.conll')
utterances = pd.concat([test_utts, valid_utts], ignore_index=True)


#--2) Find the ids of these utterances in the English split of MASSIVE
def load_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as jsonl_file:
        return [json.loads(line.strip()) for line in jsonl_file]

def find_id(utterances: pd.DataFrame, jsonl_EN_data):
    ids = []
    for index, row in utterances.iterrows():
        text_en = row['text_EN'].lower().strip()

        for line in jsonl_EN_data:
            if text_en == line.get('utt', '') or text_en == line.get('annot_utt', ''):
                ids.append(line.get('id', ''))
                break

    utterances['massive_id'] = ids
    return utterances

jsonl_EN_data = load_jsonl('data/massive/en-US.jsonl')
utterances = find_id(utterances, jsonl_EN_data)


#--3) Find Norwegian utterance given id
def find_NO_utterances(utterances: pd.DataFrame, jsonl_NO_data):
    utterances_NO = []

    for index, row in utterances.iterrows():
        massive_id = row['massive_id']

        for line in jsonl_NO_data:
            if massive_id == line.get('id', ''):
                utterances_NO.append(line.get('utt', ''))
    
    utterances['text_NO'] = utterances_NO
    return utterances

jsonl_NO_data = load_jsonl('data/massive/nb-NO.jsonl')
utterances = find_NO_utterances(utterances, jsonl_NO_data)
print(utterances)


#--4) Create Norwegian conll file
def create_new_conll_file(utterances: pd.DataFrame, output_file: str):
    with open(output_file, 'w') as f:
        for index, row in utterances.iterrows():

            #Write metadata as comments
            f.write(f"# id: {row['id']}\n")
            f.write(f"# intent: {row['intent']}\n")
            f.write(f"# intent-massive: {row['intent_massive']}\n")
            f.write(f"# text-en: {row['text_EN']}\n")
            f.write(f"# text: {row['text_NO']}\n")

            #Split annot_text_NO into tokens 
            tokens = row['text_NO'].split()
            for i, token in enumerate(tokens):
                f.write(f"{i+1}\t{token}\t{row['intent']}\t\n")
            
            #Write an empty line between utterances
            f.write("\n")

create_new_conll_file(utterances, 'data/mas.conll')