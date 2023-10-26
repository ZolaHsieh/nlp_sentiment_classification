import torch, transformers
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self,
				vocab_len,
				emb_dim,
				hidden_dim,
				n_layers,
				out_len):

        super().__init__()
        self.M = hidden_dim
        self.L = n_layers

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_len,
                                            embedding_dim=emb_dim)

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True)

        self.linear = nn.Linear(hidden_dim, out_len)

    def forward(self, x):
        # initial hidden states  L x N x M
        h0 = torch.rand(self.L, x.size(0), self.M)
        c0 = torch.rand(self.L, x.size(0), self.M)

        embeddings = self.embedding_layer(x)
        output, _ = self.lstm(embeddings, (h0, c0))

        return self.linear(output[:,-1]) # we only want h(T) at the final time step


class LSTMGloveModel(nn.Module):
  def __init__(self,
               emb_dim,
               hidden_dim,
               n_layers,
               out_len,
               weight):

    super().__init__()

    self.M = hidden_dim
    self.L = n_layers

    self.embedding_layer = nn.Embedding.from_pretrained(weight)
    self.embedding_layer.weight.requires_grad = True

    self.lstm = nn.LSTM(input_size=emb_dim,
                        hidden_size=hidden_dim,
                        num_layers=n_layers,
                        batch_first=True)

    self.linear = nn.Linear(hidden_dim, out_len)

  def forward(self, x):
    # initial hidden states  L x N x M
    h0 = torch.rand(self.L, x.size(0), self.M)
    c0 = torch.rand(self.L, x.size(0), self.M)

    embeddings = self.embedding_layer(x)
    output, _ = self.lstm(embeddings, (h0, c0))

    return self.linear(output[:,-1]) # we only want h(T) at the final time step


class BiLSTMModel(nn.Module):
  def __init__(self,
               vocab_len,
               emb_dim,
               hidden_dim,
               n_layers,
               out_len):

    super().__init__()
    self.M = hidden_dim
    self.L = n_layers

    self.embedding_layer = nn.Embedding(num_embeddings=vocab_len,
                                        embedding_dim=emb_dim)

    self.lstm = nn.LSTM(input_size=emb_dim,
                        hidden_size=hidden_dim,
                        num_layers=n_layers,
                        bidirectional=True,
                        batch_first=True)

    self.linear = nn.Linear(2*hidden_dim, out_len)

  def forward(self, x):
    # initial hidden states  L x N x M
    h0 = torch.rand(self.L*2, x.size(0), self.M)
    c0 = torch.rand(self.L*2, x.size(0), self.M)

    embeddings = self.embedding_layer(x)
    output, _ = self.lstm(embeddings, (h0, c0))

    return self.linear(output[:,-1]) # we only want h(T) at the final time step
  

class DistilBertModel(nn.Module):
  def __init__(self, num_labels):
    super().__init__()

    self.base_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
    self.pre_classifier = torch.nn.Linear(768, 768)
    self.dropout = torch.nn.Dropout(0.3)
    self.classifier = torch.nn.Linear(768, num_labels)

  def forward(self, input_ids, attention_mask):

    output = self.base_model(input_ids=input_ids,
                             attention_mask=attention_mask)

    hidden_state = output[0]
    pooler = hidden_state[:, 0]
    pooler = self.pre_classifier(pooler)
    pooler = torch.nn.Tanh()(pooler)
    pooler = self.dropout(pooler)
    output = self.classifier(pooler)

    return output