import torch
from torch import nn

class CustomBERTModel(nn.Module):
    """
    simple transformer classifier using a pretrained BERT model as base
    param pretrained_model: pretrained BERT model
    param additional_features: number of additional features besides text
    """
    def __init__(self, pretrained_model, additional_features):
          super(CustomBERTModel, self).__init__()
          self.bert = pretrained_model

          self.linear1 = nn.Linear(768 + additional_features, 256)
          self.linear2 = nn.Linear(256, 2)
          self.sig = torch.nn.Sigmoid()

    def forward(self, ids, mask, additional):
          sequence_output = self.bert(ids, attention_mask=mask).last_hidden_state
          sequence_output = torch.cat([sequence_output[:,0,:], additional], dim=1)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output) 
          linear2_output = self.linear2(linear1_output)
          linear2_output = self.sig(linear2_output)
          return linear2_output