from word_swapping import update_spelling


def update_spellings(file):
    with open(file, "r") as f:
        lines = f.readlines()

    updated_lines = []

    for line in lines:
        if line.strip() == "" or line.startswith("#"):
            updated_lines.append(line)
        
        else:
            columns = line.split("\t")
            dialect_word = columns[1]
            corrected_word = update_spelling(dialect_word)

            if corrected_word != dialect_word:
                columns[1] = corrected_word
                print(dialect_word, corrected_word)
            
                updated_line = "\t".join(columns)
                updated_lines.append(updated_line)
            
            else:
                updated_lines.append(line)

    with open(file, "w") as f:
        f.writelines(updated_lines)


train_file = 'data/aux_data/lia_train.conll'
update_spellings(train_file)

dev_file = 'data/aux_data/lia_train.conll'
update_spellings(dev_file)
