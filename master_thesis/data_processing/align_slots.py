from simalign import SentenceAligner
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import sys

# Load the XLM-R model and tokenizer
model_name = "xlm-roberta-large" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def parse_conll(file_path):
    sentences = []
    intents = []
    slots = []
    
    with open(file_path, 'r') as f:
        current_tokens = []
        current_slots = []
        current_intent = None
        for line in f:
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

def create_dataframe(en_file, nb_file):
    # Parse the gold and predicted files
    en_sentences, en_intents, en_slots = parse_conll(en_file)
    nb_sentences, nb_intents, nb_slots = parse_conll(nb_file)

    # Check for alignment
    if len(en_sentences) != len(nb_sentences):
        raise ValueError("Mismatch in number of sentences between en and nb files.")

    # Create a DataFrame
    data = {
        'en_sentence': en_sentences,  
        'nb_sentence': nb_sentences,  
        'en_intent': en_intents,
        'nb_intent': nb_intents,
        'en_slots': en_slots,
        'nb_slots': nb_slots
    }

    df = pd.DataFrame(data)

    # Only keep unique english sentences
    unique_df = df.drop_duplicates(subset='en_sentence')

    # Reset the index of the new DataFrame
    unique_df.reset_index(drop=True, inplace=True)
    return unique_df

def token_embedding(word, sentence, context_window=2):
    sentence = sentence.split()
    word_tokens = word.split()

    word_idx = sentence.index(word_tokens[0])        

    #Get embedding of context window, not just the token
    start_idx = max(0, word_idx - context_window)
    end_idx = min(len(sentence), word_idx + (len(word_tokens)-1) + context_window + 1)
    context_tokens = sentence[start_idx:end_idx]

    #Join the context tokens into a context sentence
    context_sentence = " ".join(context_tokens)

    inputs = tokenizer(context_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    #Find the embedding corresponding to the target word within the context
    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    indices = [
        idx for idx, token_id in enumerate(inputs.input_ids[0]) 
        if token_id in word_tokens
    ]
    
    #Average the embeddings of the target token and its subword tokens if necessary
    embeddings = outputs.last_hidden_state[0][indices]
    word_embedding = embeddings.mean(dim=0)  #Average over subword tokens if necessary
    
    return word_embedding

def find_most_similar_en_token(no_token, en_tokens, no_sentence, en_sentence):
    no_embedding = token_embedding(no_token, no_sentence)
    max_sim = -1
    most_similar_token = None
    
    for token in en_tokens:  #token is a tuple of token and slot
        en_embedding = token_embedding(token[0], en_sentence)
        sim = torch.cosine_similarity(no_embedding, en_embedding, dim=0).item()

        if sim > max_sim:
            max_sim = sim
            most_similar_token = token
    
    return most_similar_token

aligner = SentenceAligner(model="xlm-roberta-large", token_type="bpe", matching_methods="mai")      # Initialize the aligner
def get_new_norwegian_slots(en_sentence, no_sentence, english_slots):
    # Get word alignments
    alignment = aligner.get_word_aligns(en_sentence, no_sentence)

    # Mapping slots from English to Norwegian based on alignment
    no_to_en = dict()  #Shows if multiple English tokens are mapped to the same Norwegian token


    #1) Get initial token mappings from simAlign
    for align in alignment['mwmf']:
        en_index, no_index = align

        en_token = en_sentence[en_index]
        en_slot = english_slots[en_index]
        no_token = no_sentence[no_index]

        #Remove BIO tag from slot
        if en_slot != 'O':
            en_slot = en_slot.split('-')[1]
        
        if no_token not in no_to_en:
            no_to_en[no_token] = []
        no_to_en[no_token].append((en_token, en_slot))


    #2) Make sure a Norwegian token is only mapped to one English token (if >1, the one with highest cos similarity is chosen)
    for no_token, aligned_en_tokens in no_to_en.items():
        #If there is only one slot type for all English tokens, just map the the Norwegian token to this slot
        if len(aligned_en_tokens) == 1:
            no_to_en[no_token] = aligned_en_tokens[0]


        elif len(aligned_en_tokens) > 1:
            en_slots = [slot for token, slot in aligned_en_tokens]

            #If there is only one slot type for all English tokens, just map the the Norwegian token to this slot
            if len(set(en_slots)) == 1:  
                no_to_en[no_token] = (aligned_en_tokens[0][0], en_slots[0])  #Just maps the Norwegian token to the first English token
            
            #More than 1 possible slot to map to
            else:
                #Compare the cosine similarity between each no-en token pair, and choose the slot of the English token with highest cosine similarity
                most_similar_en_token = find_most_similar_en_token(no_token, aligned_en_tokens, ' '.join(no_sentence), ' '.join(en_sentence))
                no_to_en[no_token] = most_similar_en_token  


    #3) Add BIO tags and remove leading prepositions from slot spans
    prepositions = {'på', 'for', 'å', 'til', 'at', 'fra'}  # Add more prepositions if needed

    #print(no_to_en)
    #print(no_sentence)
    slots = []
    previous_slot = 'O'
    for i, no_token in enumerate(no_sentence):
        if no_token in no_to_en.keys():
            slot = no_to_en[no_token][1]
        
        #If a Norwegian token is not aligned with an English token, just add O
        else:
            slot = 'O'

        #Remove leading prepositions in a new slot span
        if no_token in prepositions and slot != previous_slot:
            slots.append('O')
            previous_slot = 'O'

        else:
            if slot == 'O':
                #If we just ended a slot span, check if its last token was a preposition
                if previous_slot != 'O' and i > 0 and no_sentence[i - 1] in prepositions:
                    slots[-1] = 'O'  #Convert the last token of the previous span to 'O'

                slots.append('O')
                previous_slot = 'O'
            
            #Beginning of a new slot span
            elif previous_slot != slot:  #To avoid consecutive B-slots
                #If we just ended a slot span, check if its last token was a preposition
                if previous_slot != 'O' and i > 0 and no_sentence[i - 1] in prepositions:
                    slots[-1] = 'O'  # Convert the last token of the previous span to 'O'

                slots.append(f'B-{slot}')
                previous_slot = slot
            
            #Continuing the same slot span with I-tag
            else:  #If the previous slot is the same as the current slot, and not 'O'
                slots.append(f'I-{slot}')
                previous_slot = slot 
    
    #Final check in case the last token in the sentence ends a slot span and is a preposition
    if previous_slot != 'O' and (no_sentence[-1] in prepositions or no_sentence[-1] in ['.', '?', '!', ',']):
        slots[-1] = 'O'  #Convert the last token of the final span to 'O'


    print(en_sentence)
    print(no_sentence)
    print(slots)
    print()
    sys.stdout.flush()

    return slots

def re_align_slots(conn_align_from, conll_align_to, new_conll_out):
    df = create_dataframe(conn_align_from, conll_align_to)

    new_slots = []
    for i, r in df.iterrows():
        en_sentence = r['en_sentence']
        nb_sentence = r['nb_sentence']
        en_slots = r['en_slots']
        new_slots.append(get_new_norwegian_slots(en_sentence, nb_sentence, en_slots))
        print()

    df['aligned_slots'] = new_slots

    with open(conll_align_to, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    current_sentence_tokens = []
    current_slots = []

    for line in lines:
        if line.startswith("# text ="):
            # Capture the Norwegian sentence tokens from the CoNLL line
            current_sentence_tokens = line.strip().split(" = ")[1].split()
            
            # Find the aligned slots from the DataFrame for this sentence
            matching_row = df[df['nb_sentence'].apply(lambda x: x == current_sentence_tokens)]
            
            # If a match is found, get the aligned slots
            if not matching_row.empty:
                current_slots = matching_row.iloc[0]['aligned_slots']
            else:
                current_slots = ["O"] * len(current_sentence_tokens)  # Default slots if no match found
                
            updated_lines.append(line)

        elif line.startswith("#") or line.strip() == "":  
            # Append comment lines and empty lines
            updated_lines.append(line)

        else:
            tokens = line.strip().split('\t')
            
            # Ensure the line has the expected columns
            index, token, intent, slot = tokens

            # Find the position of the token in the current sentence tokens
            token_index = current_sentence_tokens.index(token)
            new_slot = current_slots[token_index]


            # Append the modified line with the new slot
            updated_lines.append(f"{index}\t{token}\t{intent}\t{new_slot}\n")

        # Add a blank line after each complete utterance
        if line.strip() == "":
            updated_lines.append("")  


    # Write the updated lines to the new CoNLL file
    with open(new_conll_out, 'w') as file:
        for updated_line in updated_lines:
            file.write(updated_line)


re_align_slots('../data/en.conll', '../data/nb.conll', '../data/nb_ra.conll')




def fix_issues_with_bio_scheme(input_file, output_file):
    lines = []
    i = 0

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith("#"): 
                previous_tag = 'O'
                lines.append(line)
            
            elif line == "":
                previous_line = lines[-1]
                columns = previous_line.split("\t")
                token, tag = columns[1], columns[3]

                if token in [".", "!", "?"] and tag != "O":
                    columns[3] = "O"  # Change tag to O
                    lines[-1] = "\t".join(columns)
                    i += 1
                
                previous_tag = 'O'
                lines.append(line)

            else:
                columns = line.split("\t")
                token, tag = columns[1], columns[3]

                # Correct I-* following O
                if tag.startswith("I-") and previous_tag == "O":
                    correct_tag = 'B-' + tag[2:]
                    columns[3] = correct_tag
                    line = "\t".join(columns)
                    lines.append(line)
                    i += 1
                
                # Correct B-* following I-* with the same slot
                elif tag.startswith("B-") and previous_tag.startswith("I-") and tag[2:] == previous_tag[2:]:
                    correct_tag = 'I-' + tag[2:]
                    columns[3] = correct_tag
                    line = "\t".join(columns)
                    lines.append(line)
                    i += 1

                else:
                    lines.append(line)
                
                previous_tag = tag
    print(i)

    # Write the updated lines to the new CoNLL file
    with open(output_file, 'w') as file:
        for updated_line in lines:
            file.write(updated_line)
            file.write('\n')


fix_issues_with_bio_scheme("../data/nb_ra.conll", "../data/nb_ra.conll")


    
