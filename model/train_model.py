import torch
import numpy as np
from transformer import *
import pandas as pd
import transformers as ppb
from transformers import AdamW
from utils import *
from transformers import get_linear_schedule_with_warmup
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from parameters import params


def train(dataloader, device, model, criterion, optimizer, scheduler):
    """
    performs a single training loop
    param dataloader: dataloader containing (input_ids, input_mask, additional_features, labels)
    param model: model object
    param criterion: criterion to determine the loss
    param optimizer: optimizer to update the parameters
    param scheduler: learning rate scheduler

    return: avg_loss and avg_accuracy
    """
    
    total_loss, total_acc = 0, 0

    for step, batch in enumerate(dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_add = batch[2].to(device)
        b_labels = batch[3].to(device).flatten()

        # clear previous gradients
        model.zero_grad()     

        # forward pass
        logits = model(ids=b_input_ids, mask=b_input_mask, additional=b_add)   

        # calculate the loss
        loss = criterion(input=logits.float(), target=b_labels)

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        total_loss += loss.item()
        total_acc += flat_accuracy(logits, label_ids)

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update the learning rate.
        scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_acc / len(dataloader)

    return avg_loss, avg_accuracy


def validation(dataloader, device, model, criterion):
    """
    performs a single validation loop
    param dataloader: dataloader containing (input_ids, input_mask, additional_features, labels)
    param model: model object
    param criterion: criterion to determine the loss

    return: avg_loss and avg_accuracy
    """
    
    total_loss, total_acc = 0, 0

    for step, batch in enumerate(dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_add = batch[2].to(device)
        b_labels = batch[3].to(device).flatten()

        # clear previous gradients
        model.zero_grad()     

        with torch.no_grad():
            # forward pass
            logits = model(ids=b_input_ids, mask=b_input_mask, additional=b_add)   

            # calculate the loss
            loss = criterion(input=logits.float(), target=b_labels)

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        total_loss += loss.item()
        total_acc += flat_accuracy(logits, label_ids)

        # Perform a backward pass to calculate the gradients.
        loss.backward()
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_acc / len(dataloader)

    return avg_loss, avg_accuracy


def main(output_dir, dataset_dir):

    sns.set_theme()

    # prepare the output folder
    output_dir = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(f'./{output_dir}/', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the pretrained models
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    # load the dataset
    df = pd.read_csv(dataset_dir)
    df['review'] = df['review'].apply((lambda x: tokenizer.encode(' '.join(str(x).split(' ')[:130]), add_special_tokens=True)))

    # create the dataloaders
    df_train, df_valid = np.split(df.sample(frac=1, random_state=42), [int(.75*len(df))])
    train_dataloader = create_dataloader(df_train, params['batch size'])
    valid_dataloader = create_dataloader(df_valid, params['batch size'])

    model = model_class.from_pretrained(pretrained_weights)
    model = CustomBERTModel(model, len(df.columns) - 1).to(device)

    epochs = 4
    total_steps = len(train_dataloader) * epochs

    optimizer = AdamW(model.parameters(), lr=params['learning rate'], eps=params['epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for epoch in range(epochs):
        
        # training loop
        model.train()
        avg_loss, avg_accuracy =\
            train(train_dataloader, device, model, criterion, optimizer, scheduler)

        train_loss.append(avg_loss)
        train_acc.append(avg_accuracy)

        # validation loop
        model.eval()
        avg_loss, avg_accuracy = validation(valid_dataloader, device, model, criterion)

        val_loss.append(avg_loss)
        val_acc.append(avg_accuracy)
        
        # save the best model
        if avg_accuracy > best_acc:
            best_acc = avg_accuracy
            torch.save(model.state_dict(), f'./{args.output_dir}/{os.makedirs()}/transformer.MODEL')


    x = range(1, epochs+1)

    plt.plot(x, train_loss, label='train')
    plt.plot(x, val_loss, 'val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('BCE Loss')
    plt.savefig(f'./{output_dir}/{os.makedirs()}/loss.png')

    plt.plot(x, train_acc, label='train')
    plt.plot(x, val_acc, 'val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.savefig(f'./{output_dir}/{os.makedirs()}/accuracy.png')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output', type=str,
                        help='directory where the output is stored.')
    parser.add_argument('--dataset_dir', default='./review.csv', type=str,
                        help='file directory of the dataset in csv.')


    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)