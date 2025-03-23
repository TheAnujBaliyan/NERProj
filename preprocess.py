from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import pad_sequences
# from tensorflow.keras.utils import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences



def split_text_label(filename):
    split_labeled_text = []
    current_sentence = []
    
    with open(filename, 'r') as file:
        for line in file:
            # Skip document separators and empty lines
            if line.startswith('-DOCSTART') or line.strip() == '':
                if current_sentence:
                    split_labeled_text.append(current_sentence)
                    current_sentence = []
                continue
            
            # Split line into token and label
            parts = line.rstrip('\n').split()
            token = parts[0] if parts else ''
            label = parts[-1] if parts else ''
            current_sentence.append([token, label])
        
        # Add the final sentence if any
        if current_sentence:
            split_labeled_text.append(current_sentence)
    
    return split_labeled_text

def padding(sentences, labels, max_len, padding='post'):
    padded_sentences = pad_sequences(sentences, max_len, padding='post')
    padded_labels = pad_sequences(labels, max_len, padding='post')
    return padded_sentences, padded_labels

def createMatrices(data, word2idx, label2idx):
    unk = word2idx['UNKNOWN_TOKEN']
    lower_map = {k.lower(): v for k, v in word2idx.items()}
    
    return (
        [
            [word2idx.get(w, lower_map.get(w.lower(), unk)) for w, _ in sent]
            for sent in data
        ],
        [
            [label2idx[l] for _, l in sent]
            for sent in data
        ]
    )
