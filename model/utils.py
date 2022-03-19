import torch
import random
import numpy as np
from torch.utils import data


def set_seed(seed):
    """
    set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def create_dataloader(df, BATCH_SIZE, label=True):
    tokenized = df['tokenized']
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    if label:
        labels = torch.tensor(df[['label']].values)

        additional_features = []
        additional_features.append(df.drop([['review', 'label']], axis=1).to_numpy().tolist())

        dataset = list(zip(input_ids, attention_mask, additional_features, labels))
    else:
        dataset = list(zip(input_ids, attention_mask, additional_features))
    
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

    return dataloader
