#from simalign import SentenceAligner
import pandas as pd
import ast
import re


def read_conll_and_extract_sentences(file_path, output_file_path):
    unique_sentences = []
    
    # Read the file and extract unique sentences
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("# text ="):
                sentence = line.strip().split(" = ", 1)[1]
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
                else:
                    print(sentence)
    
    # Write unique sentences to the output file
    with open(output_file_path, 'w') as output_file:
        for sentence in unique_sentences:
            output_file.write(sentence + "\n")


def create_conll_from_txt(en_conll, en_txt, input_no_txt, output_no_conll, tags=False):
    """
    The input txt file is a translated version of en_txt.
    If tags=True, slot spans hace tags around them and must be handled. 
    Need therefore to:
        1) Map each Norwegian sentence to its corresponding English sentence.
        2) Go through en_conll to get all utterances (not only unique ones as in the txt files)
           and get the intents of a given utterance.
        3) Create a conll file for the Norwegian utterances (if tags are included the format of these must be changed)
    """

    def process_tagged_sentence(sentence, intent):
        #Use a regular expression to find all tagged spans
        pattern = r"<(.*?)>(.*?)</\1>"
        matches = re.finditer(pattern, sentence)

        #Store tagged spans and their labels
        tags = []
        plain_sentence = sentence  

        for match in matches:
            label = match.group(1)  
            content = match.group(2)  
            start_index = plain_sentence.find(match.group(0))  #NOTE maybe group(2)? #Start index of the tagged content
            end_index = start_index + len(content)  #End index of the tagged content

            tags.append((label, start_index, end_index, content))

            #Remove the tag from the plain sentence
            plain_sentence = plain_sentence.replace(match.group(0), content, 1)

        tokens = plain_sentence.split()

        # Map tags to tokens by finding the corresponding token positions
        tokenized_output = []
        token_start = 0

        for i, token in enumerate(tokens, start=1):
            token_end = token_start + len(token)  #To look for matches with any tag span
            token_tag = 'O'  #Default if not tagged

            #Check if the current token is part of any tagged span
            for tag in tags:
                slot_span_start, slot_span_end, label = tag[1], tag[2], tag[0]
                if token_start >= slot_span_start and token_end <= slot_span_end:
                    #Assign B- for the first token in the span, I- for subsequent tokens
                    token_tag = f"B-{tag[0]}" if token_start == slot_span_start else f"I-{tag[0]}"
                    break

            # Append tokenized output in the desired format
            tokenized_output.append(f"{i}\t{token}\t{intent}\t{token_tag}\n")
            token_start = token_end + 1  # Move to the next token position

        return tokenized_output


    #Load English-Norwegian sentence mappings
    english_to_norwegian = {}
    with open(en_txt, 'r') as en_file, \
        open(input_no_txt, 'r') as no_file:
        for en_sentence, no_sentence in zip(en_file, no_file):
            english_to_norwegian[en_sentence.strip()] = no_sentence.strip()


    #Write to conll file
    with open(en_conll, 'r') as en_file, \
        open(output_no_conll, 'w') as no_file:

        no_sent = None 

        for line in en_file:
            if line.startswith('# text ='):
                en_sent = line.split(' = ')[1].strip()
                no_sent = english_to_norwegian[en_sent]

                # Check if the last character of no_sent is punctuation and add a space before it if true (to get it as a separate token)
                if no_sent[-1] in ".,!?": 
                    no_sent = no_sent[:-1] + " " + no_sent[-1]

                if tags:
                    no_plain_sent = re.sub(r"</?[^>]+>", "", no_sent)
                    no_file.write(f'# text = {no_plain_sent}\n')
                else:
                    no_file.write(f'# text = {no_sent}\n')
                no_file.write(f'# text-en = {en_sent}\n')

            elif line.startswith('# intent = '):
                intent = line.split(' = ')[1].strip()
                no_file.write(f'# intent = {intent}\n')

                no_tokens = no_sent.split()
                
                if not tags:
                    # Write each token with "O" slots
                    for index, token in enumerate(no_tokens, start=1):
                        no_file.write(f"{index}\t{token}\t{intent}\tO\n")
                
                else:
                    tokenized_output = process_tagged_sentence(no_sent, intent)
                    for token_line in tokenized_output:
                        no_file.write(token_line)

            
            elif line.strip() == "":
                no_file.write('\n')


def add_underline_in_test_conll(file_path):
    # Read the CoNLL file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Open the output file to write the modified content
    with open(file_path, "w") as file:
        for line in lines:
            # Add underline after '# intent = '
            if line.startswith("# intent ="):
                line = line.strip() + " _\n"
            
            # Add underline after a tab at the end of each token line
            elif re.match(r"^\d+\t", line):
                line = line.strip() + "\t_\t_\n"
            
            file.write(line)


def re_align_slots(conn_align_from, conll_align_to):
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

    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")      # Initialize the aligner
    def get_new_norwegian_slots(english_sentence, norwegian_sentence, english_slots):
        # Get word alignments
        alignment = aligner.get_word_aligns(english_sentence, norwegian_sentence)

        # Initialize Norwegian slots with 'O'
        norwegian_slots = ["O"] * len(norwegian_sentence)

        # Mapping slots from English to Norwegian based on alignment
        for align in alignment['mwmf']:
            eng_index, nor_index = align
            if nor_index < len(norwegian_slots):  # Ensure the Norwegian index is valid
                english_slot = english_slots[eng_index]

                if english_slot.startswith('B-'):  
                    slot = english_slot[2:]
                    
                    if nor_index != 0:
                        last_slot = norwegian_slots[nor_index-1]

                        #Check if last slot is a B- or I- of the same slot, if so, convert to I-
                        if last_slot == f'B-{slot}' or last_slot == f'I-{slot}':  
                            norwegian_slots[nor_index] = f'I-{slot}'
                    
                        #Start a new slot with B-, if the same slot is not preceding
                        else:
                            norwegian_slots[nor_index] = english_slot
                    
                    #Start a new slot with B- if this is the first slot in the sentence
                    else:
                        norwegian_slots[nor_index] = english_slot

                elif english_slot.startswith('I'):
                    slot = english_slot[2:]
                    
                    if nor_index != 0:
                        last_slot = norwegian_slots[nor_index-1]

                        #Only add I- if it follows a B- or I- of the same type
                        if last_slot == f'B-{slot}' or last_slot == f'I-{slot}':  
                            norwegian_slots[nor_index] = english_slot

                        #Convert to B- if the preceding slot is O or has another slot
                        else:
                            norwegian_slots[nor_index] = f'B-{slot}'

                    else:
                        norwegian_slots[nor_index] = f'B-{slot}'           

                else:  #Handle 'O' by resetting last slot
                    norwegian_slots[nor_index] = 'O'
                    
        

        #Ensure no I-tag follows O
        def correct_bio_tags(slots):
            for i in range(1, len(slots)):
                if slots[i].startswith('I-') and slots[i-1] == 'O':
                    slots[i] = 'B-' + slots[i][2:]
            return slots
        norwegian_slots = correct_bio_tags(norwegian_slots)

        return norwegian_slots


    df = create_dataframe(conn_align_from, conll_align_to)

    new_slots = []
    for i, r in df.iterrows():
        en_sentence = r['en_sentence']
        nb_sentence = r['nb_sentence']
        en_slots = r['en_slots']
        new_slots.append(get_new_norwegian_slots(en_sentence, nb_sentence, en_slots))

    df['aligned_slots'] = new_slots

    """
    #If you want to save the alignments to a csv in case of errors later
    df.to_csv('df.csv', index=False)  # Set index=False to exclude the DataFrame index
    df = pd.read_csv("df.csv")

    # Convert columns with list-like strings to actual lists
    list_columns = ['en_sentence', 'nb_sentence', 'en_slots', 'nb_slots', 'aligned_slots']
    for col in list_columns:
        df[col] = df[col].apply(ast.literal_eval)
    """

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
    with open(conll_align_to, 'w') as file:
        for updated_line in updated_lines:
            file.write(updated_line)


def put_intent_in_line2(in_file, out_file):
    with open(in_file, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    output_lines = []
    instance = []
    intent = None

    for line in lines:
        # Identify the beginning of a new instance
        if line.startswith('# id ='):
            if instance:
                output_lines.extend(instance)  # Write the previous instance to output
                instance = []  # Clear for the new instance
            instance.append(line)  # Add the # id line

        elif line.startswith('# intent ='):
            intent = line.strip().split('=')[1].strip()  # Extract intent value
            instance.append(line)  # Append the original intent line
            instance.insert(1, f"# intent = {intent}\n")  # Insert # intent line after # id

        elif line.strip() == '':  # Empty line signals the end of an instance
            if instance:
                output_lines.extend(instance)
                output_lines.append('\n')  # Maintain the empty line between instances
                instance = []  # Reset for the next instance
            intent = None

        else:
            instance.append(line)  # Add the rest of the instance lines

    # Add the last instance if the file doesn't end with a blank line
    if instance:
        output_lines.extend(instance)

    # Overwrite the same file with the updated content
    with open(out_file, 'w') as file:
        file.writelines(output_lines)


def add_sentence_ids(conll_file):
    with open(conll_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    sentence_id = 1
    with open(conll_file, "w", encoding="utf-8") as output:
        for line in lines:
            if line.startswith("# text = "):  # Identify the start of a new sentence
                output.write(f"# id = {sentence_id}\n")  # Add the sentence ID
                sentence_id += 1
            output.write(line)


def fix_issues_with_bio_scheme(conll_file):
    lines = []
    i = 0

    with open(conll_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("# text ="):
                text = line.split(' = ')
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
    with open(conll_file, 'w') as file:
        for updated_line in lines:
            file.write(updated_line)
            file.write('\n')


#read_conll_and_extract_sentences('data/en.conll', 'data/test.txt')

#create_conll_from_txt('data/en.conll', 'data_processing/en.txt', 'data_processing/nn1.txt', 'data/nn1.conll', tags=True)

#re_align_slots('data/en.train.conll', 'new1.nb.xsid.train.conll')

#add_underline_in_test_conll("norsid_test_nolabels.conll")

#put_intent_in_line2('data/norsid_dev.conll', 'data/norsid_dev_changed.conll')
    
#add_sentence_ids('data/corrected_new1_nb.conll')


#input_file = "data/new1_nb.conll"
#output_file = "data/corrected_new1_nb.conll"
#fix_issues_with_bio_scheme('data/nn.conll')