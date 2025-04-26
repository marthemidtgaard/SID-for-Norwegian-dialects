import random

def noisy_indices(sent_toks, percentage_noisy, seed):
    random.seed(seed)

    # Only include words with alphabetic content
    poss_indices = [i for i, tok in enumerate(sent_toks)
                    if any(c.isalpha() for c in tok)]
    idx_noisy = random.sample(
        poss_indices, k=round(percentage_noisy * len(poss_indices)))
    return idx_noisy


def add_char(word, seed):
    random.seed(seed)

    alphabet = 'abcdefghijklmnopqrstuvwxyzæøå'
    idx = random.randrange(-1, len(word))
    added_char = random.sample(alphabet, 1)[0]

    if idx == -1: # Beginning of word
        if word[0].isupper():
            return added_char.upper() + word
        return added_char + word
    return word[:idx + 1] + added_char + word[idx + 1:]


def delete_char(word, seed):
    random.seed(seed)

    idx = random.randrange(len(word))
    return word[:idx] + word[idx + 1:]


def replace_char(word, seed):
    random.seed(seed)

    alphabet = 'abcdefghijklmnopqrstuvwxyzæøå'
    idx = random.randrange(len(word))
    replaced_char = random.sample(alphabet, 1)[0]

    if idx == 0 and word[0].isupper():
        replaced_char = replaced_char.upper()  

    return word[:idx] + replaced_char + word[idx + 1:]

def swap_chars(word, seed):
    random.seed(seed)
    
    if len(word) < 2:
        return word
    
    idx = random.randrange(len(word) - 1)
    if idx == 0 and word[0].isupper():
        return word[1].upper() + word[0].lower() + word[2:]

    return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]


def randomly_choose_noise(words, noise_lvl, seed):
    random.seed(seed)
    noise_operations = (add_char, delete_char, replace_char, swap_chars)
    n_changed = 0
    idx_noisy = sorted(noisy_indices(words, noise_lvl, seed))
    words_noisy = []
    for i, word in enumerate(words):
        if i in idx_noisy:
            noise_type = random.choice(noise_operations)
            word_noised = noise_type(word, seed)
            words_noisy.append(word_noised)
            n_changed += 1
    return idx_noisy, words_noisy


#Random noise operations
def add_random_noise(in_file, out, seed):
    random.seed(seed)

    #for noise_lvl in [0.25, 0.5, 0.75, 1.0]:
    for noise_lvl in [0.25, 0.5, 0.75]:
        lines = []
        with open(in_file) as f_in:
            sentence = []
            for line in f_in:
                if line[0] == "#":
                    lines.append(line)
                    sentence = []
                elif not line.strip():
                    if sentence:
                        words = [token_details[1] for token_details in sentence]
                        idx_noisy, words_noisy = randomly_choose_noise(words, noise_lvl, seed)
                        for i, word in zip(idx_noisy, words_noisy):
                            sentence[i][1] = word
                        for token_details in sentence:
                            lines.append("\t".join(token_details))
                    sentence = []
                    lines.append(line)
                else:
                    sentence.append(line.split("\t"))
        with open(
                f"{out}_noise_{int(noise_lvl * 100):02d}.conll",
                "w+") as f_out:
            for line in lines:
                f_out.write(line)

add_random_noise("data/mas_swapped.conll", "data/noise_data/massive/mas", seed=42)
#add_random_noise("data/nb_rt1_swapped.conll", "data/noise_data/xsid/nb_rt1_swapped", seed=42)


#5 copies of each utterance, one per noise operation
def create_stacked_noise(in_file, out, seed):
    random.seed(seed)

    noise_operations = {
    "add_char": add_char,
    "delete_char": delete_char,
    "replace_char": replace_char,
    "swap_chars": swap_chars
    }
    #for noise_lvl in [0.25, 0.5, 0.75, 1.0]:
    for noise_lvl in [0.75]:
        lines = []

        with open(in_file) as f_in:
            sentence = []
            metadata = [] 
            
            for line in f_in:
                if line.startswith("#"):
                    metadata.append(line) 
                elif not line.strip():
                    if sentence:
                        words = [token_details[1] for token_details in sentence]

                        # Generate different noise versions
                        idx_noisy = noisy_indices(words, noise_lvl, seed)

                        all_versions = {"original": words[:]}  # Store the original version

                        # Apply each noise operation directly
                        for noise_type, noise_func in noise_operations.items():
                            noisy_words = words[:]
                            for i in idx_noisy:
                                noisy_words[i] = noise_func(noisy_words[i], seed)
                            all_versions[noise_type] = noisy_words

                        # Stack all versions of the sentence together
                        for version_type, words_noisy in all_versions.items():
                            lines.extend(metadata)
                            lines.append(f"# noise = {version_type}\n")

                            # Apply noise to the sentence and write it
                            stacked_sentence = [token_details[:] for token_details in sentence]  # Deep copy
                            for i, word in enumerate(words_noisy):
                                stacked_sentence[i][1] = word
                            for token_details in stacked_sentence:
                                lines.append("\t".join(token_details))
                            
                            lines.append("\n") 

                    # Reset sentence and metadata
                    sentence = []
                    metadata = []
                    lines.append(line)
                else:
                    sentence.append(line.split("\t"))

        output_filename = f"{out}_noise_{int(noise_lvl * 100):02d}_stacked.conll"
        with open(output_filename, "w+") as f_out:
            f_out.writelines(lines)

#create_stacked_noise("data/mas_swapped.conll", "data/noise_data/massive/mas", seed=42)
#create_stacked_noise("data/nb_rt1.conll", "data/noise_data/xsid/nb_rt1", seed=42)