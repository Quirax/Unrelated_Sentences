import json
import spacy
from tqdm import tqdm
import numpy as np

# PARAMETERS
DEBUG_MODE       = True
TRAIN_FILENAME   = 'dataset/train.json'
TEST_FILENAME    = 'dataset/test.json'
ENCODING         = 'utf-8'
STOP_WORDS       = { "'", "-", "–", "—", "―", " ", "!", "$", "%", "(", ")", ",", ".", "/", ":", ";", "?", "─", "°", "\"" }
VECTOR_SIZE      = 300
NUM_OF_SENTENCES = 6

# Sequence or Tqdm
def seq(s):
    if DEBUG_MODE:
        return tqdm(s)
    else:
        return s

# Dataset validator
def validate_dataset(dataset):
    for item in dataset:
        if len(item['sentences']) != NUM_OF_SENTENCES:
            raise 'ERROR: The number of sentences is invalid'

# Load datasets
train_fp = open(TRAIN_FILENAME, 'r', encoding=ENCODING)
test_fp  = open(TEST_FILENAME,  'r', encoding=ENCODING)
train    = json.load(train_fp)
test     = json.load(test_fp)

# Load predefined english model
nlp = spacy.load('en_core_web_lg')

nlp.Defaults.stop_words |= STOP_WORDS

if __name__ == '__main__':
    # Validate dataset
    for dataset in (train, test):
        validate_dataset(dataset)
    
        for i, item in enumerate(seq(dataset)):
            dataset[i]['tokens'] = []
            dataset[i]['vectors'] = []
            for sentence in item['sentences']:
                tokens = [tk for tk in nlp(sentence) if tk.text not in nlp.Defaults.stop_words and tk.has_vector]
                dataset[i]['tokens'].append(tokens)

                vector = sum([tk.vector for tk in tokens]) / len(tokens) if len(tokens) > 0 else np.zeros(VECTOR_SIZE)
                dataset[i]['vectors'].append(vector)