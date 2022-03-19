import numpy as np
import pandas as pd
import torch
import transformers as ppb
from torch import nn
import argparse
import transformers as ppb
from utils import *
from parameters import params
from transformer import *


def main(output_dir, dataset_dir, model_dir):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    df = pd.read_csv(dataset_dir)
    df['review'] = df['review'].apply((lambda x: tokenizer.encode(' '.join(str(x).split(' ')[:130]), add_special_tokens=True)))

    model = model_class.from_pretrained(pretrained_weights)
    model = CustomBERTModel(model).to(device)

    model.load_state_dict(torch.load(model_dir))
    model.eval()

    dataloader = create_dataloader(df, params['batch size'], label=False)

    f = open('output_dir', 'w')

    # For each batch of training data...
    for step, batch in enumerate(dataloader):
        print(step)

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_add = batch[2].to(device)

        logits = model(ids=b_input_ids, mask=b_input_mask, additional=b_add).detach().cpu().numpy()
        for i in np.argmax(logits, axis=1).flatten():
            if i == 1:
                f.write('True\n')
            else:
                f.write('False\n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', default='output.txt', type=str,
                        help='file and directory where the output is stored.')
    parser.add_argument('--dataset_dir', default='./review.csv', type=str,
                        help='file directory of the dataset in csv.')
    parser.add_argument('--model_dir', default='./transformer.MODEL', type=str,
                    help='file directory of the transformer model.')

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)