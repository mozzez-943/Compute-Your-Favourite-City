import pandas as pd
import numpy as np
import random

city_mapping = {'Dubai': 0, 'Rio de Janeiro': 1, 'New York City': 2, 'Paris': 3}  # maps city name to class index

# Gets rid of commas from numbers eg 10,000 => 10000
def clean_num(s: str):
    if isinstance(s, str):
        return s.replace(',', '')
    return s


# returns the list of numbers in the string s. Used for Q6
def get_number_list(s: str) -> list:
    acc = []
    for char in s:
        if char.isdigit():
            acc.append(char)

    if len(acc) != 6:
        return [0] * 6
    return acc

# Convert words to lower case, remove punctuations
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    v = set('abcdefghijklmnopqrstuvwxyz ')
    out = ""
    for char in s:
        if char.lower() in v:
            out += char.lower()
    return out


"""
Gets data from csv file and encodes into numerical numpy array
"""
def get_data_from_csv(filename, bow_model):
    df = pd.read_csv(filename)

    del df['id']

    # Convert Q5 to one-hot encoding
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        cat_name = f"Q5{cat}"
        df[cat_name] = df['Q5'].apply(lambda s: int(cat in s) if not pd.isna(s) else 0)

    del df['Q5']

    for idx, cat in enumerate(['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine',
            'Economic']):
        cat_name = f"Q6{cat}"
        df[cat_name] = df['Q6'].apply(lambda s: get_number_list(s)[idx])

    del df['Q6']

    df['Q7'] = df['Q7'].apply(clean_num)
    df['Q9'] = df['Q9'].apply(clean_num)


    df['Q10'] = df['Q10'].apply(clean_text)

    ts = df['Label']
    ts = ts.apply(lambda s: np.eye(4)[city_mapping[s]]) # one hot encoding
    del df['Label']


    matrix = bow_model.transform(df['Q10'].tolist())

    del df['Q10']

    new = pd.concat([df, pd.DataFrame(matrix)], axis=1)


    X = np.stack(new.to_numpy()).astype(float)
    ts = np.stack(ts.to_numpy()).astype(float)

    X = np.nan_to_num(X)  # replace nan values with 0 (there are 7 rows of data that have nan values)

    return X, ts, [0,1,2,3,4,5,6,11,12,13,14,15,16]    # returns the indices that are numerical for normalization later



def normalize_data(X, mean, std, numerical_indices):
    return (X[:,numerical_indices] - mean) / std



class BOW_Model:
    def __init__(self):
        self.word_dict = {}

    def set_word_dict(self, filename):
        quotes = pd.read_csv(filename)['Q10'].apply(clean_text).tolist()
        stop_words = ['the', 'and', 'when', 'will']
        idx = 0
        for line in quotes:
            for word in line.split():
                if len(word) <= 2:
                    continue
                if word not in self.word_dict and word not in stop_words:
                    self.word_dict[word] = idx
                    idx += 1


    """
    Converts a list of sentences into a np array in the bag of words model
    """
    def transform(self, sentence_list):
        X = np.zeros((len(sentence_list), len(self.word_dict)))
        for i, line in enumerate(sentence_list):
            for word in line.split():
                if word in self.word_dict:
                    X[i,self.word_dict[word]] = line.count(word)

        return X





class DataLoader:
    def __init__(self, dataset: np.array, labels: np.array, batch_size=16, shuffle=True):
        self.dataset = dataset
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_sequence = list(range(len(self.dataset)))
        if shuffle:
            random.shuffle(self.data_sequence)
        self.idx = 0  # index into data_sequence index array

    def __len__(self):
        return len(self.dataset)

    def get_next_batch(self):
        batch_data = []
        batch_labels = []
        for i in range(self.batch_size):
            index = self.data_sequence[self.idx]
            batch_data.append(self.dataset[index])
            batch_labels.append(self.labels[index])
            self.idx += 1
        return np.array(batch_data), np.array(batch_labels)

    # reset state variables and reshuffle data after one epoch completed
    def reset_epoch(self):
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.data_sequence)





