import json
import spacy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MODES
DEBUG_MODE       = True
USE_PICKLE       = True

# PARAMETERS
TRAIN_FILENAME   = 'dataset/train.json'
TEST_FILENAME    = 'dataset/test.json'
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
RANDOM_SEED      = 85624965 # 85624965 (Alt. 91263498)
RNN_MODEL        = nn.RNN
RNN_SIZE         = 20 # 20
RNN_LAYERS       = 1
RNN_PARAMS       = {}
FC1_SIZE         = 4 # 4
DROPOUT_RATE     = 0.2

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
        fc2 = self.fc3(fc1)
        softmax = self.softmax(fc2)
        return softmax

# Sequence or Tqdm
def seq(s, desc):
    if DEBUG_MODE:
        return tqdm(s, desc=desc)
    else:
        return s

# Dataset validator
def validate_dataset(dataset):
    for item in dataset:
        if len(item['sentences']) != NUM_OF_SENTENCES:
            raise 'ERROR: The number of sentences is invalid'

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
def evaluate(model, dataset, loss_fn):
    model.eval()
    test_loss, n_correct, n_data = 0, 0, 0

    with torch.no_grad():
        for item in dataset:
            data = torch.asarray(np.array([item['vectors']]))

            target = torch.zeros((1, NUM_OF_CHOICES), dtype=torch.float)
            target[0][item['answer'] - 1] = 1.0

            output = model(data)
            loss = loss_fn(output, target)
            predict = torch.argmax(output, dim=1)

            test_loss += loss.item()
            if predict + 1 == item['answer']: n_correct += 1
            n_data += 1
    
    return test_loss / n_data, n_correct / n_data

# Pickler
if USE_PICKLE:
    REQUIRED_SAVE = False

    import pickle

    def save_pickle(train, test):
        with open('datasets.pickle', 'wb') as f:
            pickle.dump((train, test), f)
    
    def load_pickle():
        global REQUIRED_SAVE
        try:
            with open('datasets.pickle', 'rb') as f:
                train, test = pickle.load(f)
            
            return train, test
        except:
            REQUIRED_SAVE = True
            return None, None

# Load datasets
if USE_PICKLE:
    train, test = load_pickle()
    if REQUIRED_SAVE:
        train_fp = open(TRAIN_FILENAME, 'r', encoding=ENCODING)
        test_fp  = open(TEST_FILENAME,  'r', encoding=ENCODING)
        train    = json.load(train_fp)
        test     = json.load(test_fp)

# Load predefined english model
nlp = spacy.load('en_core_web_lg')

nlp.Defaults.stop_words |= STOP_WORDS

torch.manual_seed(RANDOM_SEED)

if __name__ == '__main__':
    if not USE_PICKLE or REQUIRED_SAVE:
        # Validate dataset
        for dataset in (train, test):
            validate_dataset(dataset)
        
            for i, item in enumerate(seq(dataset, "Making vectors for each sentences")):
                dataset[i]['vectors'] = []
                
                for sentence in item['sentences']:
                    tokens = nlp(sentence)
                    dataset[i]['vectors'].append(tokens.vector)

        if USE_PICKLE: save_pickle(train, test)

    model = Model()
    target_model = model
    loss = F.cross_entropy
    optimizer = torch.optim.Adadelta(model.parameters(), **OPTIMIZER_PARAMS)

    p_train_loss, p_train_acc, p_valid_loss, p_valid_acc = 0, 0, 0, 0
    t_valid_loss, t_valid_acc = 0, 0

    for epoch in range(1, MAX_EPOCH + 1):
        train_loss, train_acc = do_train(model, train, loss, optimizer)
        valid_loss, valid_acc = evaluate(model, test, loss)

        if (valid_loss < t_valid_loss and valid_acc >= t_valid_acc) or epoch % LOG_EPOCH == 0:
            print(f"{epoch: >6} => TrLoss={train_loss:.6f}({train_loss - p_train_loss:.6f}), TrAcc={train_acc:.3f}({train_acc - p_train_acc:.3f}), " \
                f"VaLoss={valid_loss:.6f}({valid_loss - p_valid_loss:.6f}), VaAcc={valid_acc:.3f}({valid_acc - p_valid_acc:.3f})")
            if valid_loss < t_valid_loss and valid_acc >= t_valid_acc:
                target_model = model
                t_valid_loss, t_valid_acc = valid_loss, valid_acc
            elif t_valid_loss == 0:
                t_valid_loss, t_valid_acc = valid_loss, valid_acc
            p_train_loss, p_train_acc, p_valid_loss, p_valid_acc = train_loss, train_acc, valid_loss, valid_acc 