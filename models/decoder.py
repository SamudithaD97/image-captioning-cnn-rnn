import torch
import torch.nn as nn
from models.attention import Attention

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        batch_size = features.size(0)
        hidden = torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)
        cell = torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)

        outputs = []
        embeddings = self.embed(captions)

        for t in range(embeddings.size(1)):
            context, _ = self.attention(features.unsqueeze(1), hidden)
            lstm_input = torch.cat((embeddings[:, t], context), dim=1)
            hidden, cell = self.lstm(lstm_input, (hidden, cell))
            output = self.fc(hidden)
            outputs.append(output)

        return torch.stack(outputs, dim=1)