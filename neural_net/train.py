import numpy as np
from dataloader import DataLoader, BOW_Model, get_data_from_csv, normalize_data
from model import Model, OneLayer, cross_entropy_loss

TRAIN_SET = 'train_dataset.csv'
VAL_SET = 'val_dataset.csv'


class Train:
    def __init__(self, model: Model, dataset: DataLoader, val_set=None, lr=0.002):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.epochs = 100
        self.val_set = val_set

    def train(self):
        num_iters = len(self.dataset) // self.dataset.batch_size

        for epoch in range(self.epochs):
            self.dataset.reset_epoch()
            for _ in range(num_iters):
                data_batch, label_batch = self.dataset.get_next_batch()
                pred = self.model.forward(data_batch)
                loss = cross_entropy_loss(pred, label_batch)
                self.model.backward(data_batch, label_batch, self.lr)


            val_accuracy = self.inference(self.val_set)
            train_accuracy = self.inference(self.dataset)
            print(f'Epoch: {epoch}, Training loss: {loss}, val accuracy: {val_accuracy*100}%, train accuracy: {train_accuracy*100}%')


    """
    Runs inference on test_set and outputs the accuracy (num_correct / total)
    """
    def inference(self, test_set: DataLoader):
        test_set.reset_epoch()
        num_iters = len(test_set) // test_set.batch_size
        num_correct = 0
        for _ in range(num_iters):
            data_batch, label_batch = test_set.get_next_batch()
            pred = self.model.forward(data_batch)
            pred_label_one_hot = np.eye(4)[np.argmax(pred, axis=1)]
            num_correct += (pred_label_one_hot == label_batch).all(axis=1).sum()
        return num_correct / len(test_set)





if __name__ == '__main__':
    bow_model = BOW_Model()
    bow_model.set_word_dict(TRAIN_SET)

    X_train, t_train, numerical_indices = get_data_from_csv(TRAIN_SET, bow_model)

    mean = X_train[:,numerical_indices].mean(axis=0)
    std = X_train[:,numerical_indices].std(axis=0)
    X_train[:,numerical_indices] = normalize_data(X_train, mean, std, numerical_indices)

    X_val, t_val, numerical_indices = get_data_from_csv(VAL_SET, bow_model)
    X_val[:,numerical_indices] = normalize_data(X_val, mean, std, numerical_indices)


    dataset = DataLoader(X_train, t_train, batch_size=4)
    print('D:', X_train.shape[1])
    model = OneLayer(X_train.shape[1], 4)

    val_set = DataLoader(X_val, t_val, batch_size=1)

    train = Train(model, dataset, val_set=val_set)
    train.train()

