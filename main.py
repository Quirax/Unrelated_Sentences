import json
import spacy

# PARAMETERS
DEBUG_MODE     = False
TRAIN_FILENAME = 'dataset/train.json'
TEST_FILENAME  = 'dataset/test.json'
ENCODING       = 'utf-8'

# Dataset validator
def validate_dataset(dataset):
    for item in dataset:
        if DEBUG_MODE:
            print(f"## {item['comment']}")
            print(f"- Number of sentences: {len(item['sentences'])}")
            print(f"- Answer: {item['answer']}")

        if len(item['sentences']) != 6:
            raise 'ERROR: The number of sentences is invalid'

# Load datasets
train_fp = open(TRAIN_FILENAME, 'r', encoding=ENCODING)
test_fp  = open(TEST_FILENAME,  'r', encoding=ENCODING)
train    = json.load(train_fp)
test     = json.load(test_fp)

# Load predefined english model
nlp = spacy.load('en_core_web_lg')

if __name__ == '__main__':
    # Validate dataset
    for dataset in (train, test):
        validate_dataset(dataset)
        if DEBUG_MODE: print()
