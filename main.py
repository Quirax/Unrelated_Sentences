import json
import spacy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pickle

# MODES
DEBUG_MODE       = True
USE_PICKLE       = True

# PARAMETERS
TRAIN_FILENAME   = 'dataset/train.json'
TEST_FILENAME    = 'dataset/test.json'
QUERY_FILENAME   = 'query.json'
ENCODING         = 'utf-8'
VECTOR_SIZE      = 300
NUM_OF_SENTENCES = 6
NUM_OF_CHOICES   = 5

STOP_WORDS       = {
    "'", "-", "–", "—", "―", " ", "!", "$", "%", "(", ")", ",", ".", "/", ":", ";", "?", "─", "°", "\"",
}

OPTIMIZER_PARAMS = {'lr': 1} # lr = 1
MAX_EPOCH        = 20000
LOG_EPOCH        = 100
RANDOM_SEED      = 3484336315 # 3484336315 (Alt. 85624965)
RNN_MODEL        = nn.RNN
RNN_SIZE         = 20 # 20
RNN_LAYERS       = 1
RNN_PARAMS       = {}
FC1_SIZE         = 4 # 4

# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.rnn = RNN_MODEL(VECTOR_SIZE, RNN_SIZE, RNN_LAYERS, batch_first=True, **RNN_PARAMS)
        self.fc1 = nn.Linear(RNN_SIZE, FC1_SIZE)
        self.fc2  = nn.Linear(FC1_SIZE, NUM_OF_CHOICES)
        self.softmax = nn.Softmax(dim=1)

        self.fc2.weight.data.uniform_(-2.0, 2.0)
    
    def forward(self, vectors):
        hidden = None
        for v in vectors[0]:
            outputs, hidden = self.rnn(v.unsqueeze(0).unsqueeze(0), hidden)
        fc1 = outputs.squeeze(0)
        fc1 = self.fc1(fc1)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        softmax = self.softmax(fc2)
        return softmax

# Output mode
def seq(s, desc):
    if DEBUG_MODE:
        return tqdm(s, desc=desc, file = sys.stderr)
    else:
        return s

def pprint(s):
    if DEBUG_MODE:
        print(s, file=sys.stderr)

# Dataset preprocessors
def validate_dataset(dataset):
    for item in dataset:
        if len(item['sentences']) != NUM_OF_SENTENCES:
            raise 'ERROR: The number of sentences is invalid'

def preprocessor(dataset, nlp):
    validate_dataset(dataset)
        
    for i, item in enumerate(seq(dataset, "Making vectors for each sentences")):
        dataset[i]['vectors'] = []
        
        for sentence in item['sentences']:
            tokens = nlp(sentence)
            dataset[i]['vectors'].append(tokens.vector)

# Trainer
def do_train(model, dataset, loss_fn, optimizer):
    model.train()
    train_loss, n_correct, n_data = 0, 0, 0

    for item in dataset:
        data = torch.asarray(np.array([item['vectors']]))
        
        target = torch.zeros((1, NUM_OF_CHOICES), dtype=torch.float)
        target[0][item['answer'] - 1] = 1.0

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        predict = torch.argmax(output, dim=1)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if predict + 1 == item['answer']: n_correct += 1
        n_data += 1
    
    return train_loss / n_data, n_correct / n_data

# Evaluater
def evaluate(model, dataset, loss_fn, doPrint=False):
    def eprint(s):
        if doPrint: print(s)

    model.eval()
    test_loss, n_correct, n_data = 0, 0, 0

    with torch.no_grad():
        for i, item in enumerate(dataset):
            eprint(f"")
            for j, s in enumerate(item['sentences']):
                if j == 0: eprint(f"{i+1:>3}. {s}")
                else: eprint(f" ({j}) {s}")
            
            data = torch.asarray(np.array([item['vectors']]))
            output = model(data)
            predict = torch.argmax(output, dim=1)

            eprint(f"Predict: {predict[0] + 1}")

            if item['answer']:
                target = torch.zeros((1, NUM_OF_CHOICES), dtype=torch.float)
                target[0][item['answer'] - 1] = 1.0
                loss = loss_fn(output, target)

                eprint(f"Actual Answer: {item['answer']}")

                test_loss += loss.item()
                if predict + 1 == item['answer']: n_correct += 1
                n_data += 1
            
            eprint(f"\n{'-'*10}\n")
    
    return test_loss / n_data, n_correct / n_data

# Pickler
REQUIRED_SAVE_MODEL = False

def save_model(model):
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)

def load_model():
    global REQUIRED_SAVE_MODEL
    try:
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
        
        return model
    except:
        REQUIRED_SAVE_MODEL = True
        return None

if USE_PICKLE:
    REQUIRED_SAVE_DATASETS = False
    
    def save_datasets(train, test):
        with open('datasets.pickle', 'wb') as f:
            pickle.dump((train, test), f)
    
    def load_datasets():
        global REQUIRED_SAVE_DATASETS
        try:
            with open('datasets.pickle', 'rb') as f:
                train, test = pickle.load(f)
            
            return train, test
        except:
            REQUIRED_SAVE_DATASETS = True
            return None, None

# Load pickles
if USE_PICKLE:
    train, test = load_datasets()
    if REQUIRED_SAVE_DATASETS:
        train_fp = open(TRAIN_FILENAME, 'r', encoding=ENCODING)
        test_fp  = open(TEST_FILENAME,  'r', encoding=ENCODING)
        train    = json.load(train_fp)
        test     = json.load(test_fp)
    
    model = load_model()

# Load predefined english model
nlp = spacy.load('en_core_web_lg')

nlp.Defaults.stop_words |= STOP_WORDS

torch.manual_seed(RANDOM_SEED)

query_fp = open(QUERY_FILENAME,  'r', encoding=ENCODING)
query    = json.load(query_fp)

if __name__ == '__main__':
    if not USE_PICKLE or REQUIRED_SAVE_DATASETS:
        # Validate dataset
        for dataset in (train, test):
            preprocessor(dataset, nlp)

        if USE_PICKLE: save_datasets(train, test)

    loss = F.cross_entropy

    if not USE_PICKLE or REQUIRED_SAVE_MODEL:
        model = Model()
        optimizer = torch.optim.Adadelta(model.parameters(), **OPTIMIZER_PARAMS)

        t_valid_loss, t_valid_acc, t_epoch = 0, 0, 0

        for epoch in seq(range(1, MAX_EPOCH + 1), "Training model"):
            do_train(model, train, loss, optimizer)
            valid_loss, valid_acc = evaluate(model, test, loss)

            if (valid_loss < t_valid_loss and valid_acc >= t_valid_acc) or t_valid_loss == 0:
                t_epoch, t_valid_loss, t_valid_acc = epoch, valid_loss, valid_acc
                save_model(model)
        
        model = load_model()
        pprint(f"Using model obtained at epoch={t_epoch} with VaLoss={t_valid_loss:.3f}, VaAcc={t_valid_acc:.3f}")

    preprocessor(query, nlp)
    query_loss, query_acc = evaluate(model, test, loss, doPrint=True)
    pprint(f"Query Loss={query_loss:.3f}, Acc={query_acc:.3f}")