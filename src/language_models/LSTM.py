# code inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, num_tokens, emb_size, hidden_size, num_layers=1, p_drop=0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p_drop)
        self.embedding = nn.Embedding(num_tokens, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=p_drop,
                            bidirectional=bidirectional,
                            batch_first=True)
        if bidirectional:
            in_features = 2 * hidden_size
        else:
            in_features = hidden_size
        self.fc = nn.Linear(in_features=in_features, out_features=num_tokens)

    def forward(self, input):
        emb = self.embedding(input)  # (B, seq_len, emb_size)
        emb = self.dropout(emb)
        output, hidden = self.lstm(
            emb)  # output (B, seq_len, hidden_size*num_dimension) # hidden: (num_layers * num_directions, B, hidden_size)
        output = self.dropout(output)
        logits = self.fc(output)  # (S,B,num_tokens)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        log_probas = F.log_softmax(logits, dim=-1)
        return log_probas, logits


if __name__ == '__main__':
    batch_size = 8
    emb_size = 512
    hidden_size = 128
    num_tokens = 85
    seq_len = 5
    device = torch.device("cpu")
    inputs = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

    # ------------ Test of LSTM Model ---------------------------------------------------------------------------------------------------------------------------

    model = LSTMModel(num_tokens=num_tokens, emb_size=emb_size, hidden_size=hidden_size, bidirectional=True)
    output, (h, c) = model(inputs)
    print('output', output.shape)
    print('hidden state', h.shape)
    print('cell state', c.shape)

    # ----------- Test of LSTM with LayerNorm Model -------------------------------------------------------------------------------------------------------------

    model = LayerNormLSTMModel(num_tokens=num_tokens, emb_size=emb_size, hidden_size=hidden_size, num_layers=2,
                               p_drop=1)
    output, (h, c) = model(inputs)
    print('output', output.shape)
    print('hidden state', h.shape)
    print('cell state', c.shape)
