import pandas as pd
import numpy as np
import random
import dataloader

file_name = "clean_dataset.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    number = ''
    for character in str(s):
        if character.isdigit():
            number += character
        elif number:
            yield int(number)
            number = ''
    if number:
        yield int(number)

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """

    n_list = list(get_number_list(s))
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = list(get_number_list(s))
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def all_float_to_int(l):
    """Convert all floats in a list to integers.
    """
    return [int(x) if isinstance(x, float) else x for x in l]

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # Clean numerics

    df["Q7"] = df["Q7"].apply(to_numeric).fillna(0)

    # Clean for number categories

    df["Q1"] = df["Q1"].apply(get_number)

    # Create area rank categories

    df["Q6"] = df["Q6"].apply(get_number_list_clean)

    # change Q1 to Q4 and Q8 to integers from floats
    for col in ["Q1", "Q2", "Q3", "Q4", "Q8"]:
        df[col] = df[col].apply(lambda x: int(x) if not pd.isna(x) else 0)

    temp_names = []
    for i in range(1,7):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))

    del df["Q6"]

    # Create category indicators

    new_names = []
    for col in ["Q1"] + temp_names:
        indicators = pd.get_dummies(df[col], prefix=col)
        new_names.extend(indicators.columns)
        df = pd.concat([df, indicators], axis=1)
        del df[col]

    # Create multi-category indicators

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"Q5{cat}"
      new_names.append(cat_name)
      df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    del df["Q5"]

    # Prepare data for training - use a simple train/test split for now

    df = df[new_names + ["Q7", "Label"]]

    # Convert the DataFrame to a list of lists (each row is a list)
    D = df.values.tolist()

    # Set the random seed and shuffle the dataset
    random.seed(100)
    random.shuffle(D)

    df2 = pd.DataFrame(D, columns=df.columns)

    df2 = df2.sample(frac=1, random_state=random_state)

    x = df2.drop("Label", axis=1).values
    y = pd.get_dummies(df2["Label"].values)

    n_train = 1175
    n_val = 147

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_val = x[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    x_test = x[n_train + n_val:]
    y_test = y[n_train + n_val:]

    # save to csv files
    train_df = pd.concat([pd.DataFrame(x_train, columns=df2.columns[:-1]), y_train], axis=1)
    train_df.to_csv('train.csv', index=False)

    valid_df = pd.concat([pd.DataFrame(x_val, columns=df2.columns[:-1]), y_val], axis=1)
    valid_df.to_csv('valid.csv', index=False)

    test_df = pd.concat([pd.DataFrame(x_test, columns=df2.columns[:-1]), y_test], axis=1)
    test_df.to_csv('test.csv', index=False)

    # Train and evaluate classifiers
    for i in range(1, 10):
        print("Testing kNN with k =", i)
        clf = dataloader.KNNModel(i)
        clf.fit(x_train, y_train)
        train_acc = clf.score(x_train, y_train)
        val_acc = clf.score(x_val, y_val)
        test_acc = clf.score(x_test, y_test)
        print(f"{type(clf).__name__} train acc: {train_acc}")
        print(f"{type(clf).__name__} validation acc: {val_acc}")
        print(f"{type(clf).__name__} test acc: {test_acc}")
        print("-------------------------")