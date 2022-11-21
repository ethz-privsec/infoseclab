import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PREFIX = "Florian's password is "


class Data:
    chars = ['\n', ' ', '!', '"', '&', "'", '(', ')', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8',
             '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
             'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '¢', '£', '¨', '©',
             'ª', '®', '°', '±', '´', '¶', '»', '¼', '½', 'Â', 'Ã', 'â', 'Ÿ', '€', '™']
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(torch.unsqueeze(input_seq, dim=-1))
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output.squeeze(1), (hidden_state[0].detach(), hidden_state[1].detach())


class LanguageModel(RNN):
    def __init__(self, ckpt_path, device):
        super().__init__(Data.vocab_size, Data.vocab_size, 512, 3)
        self.load_state_dict(torch.load(ckpt_path))
        self.to(device)
        self.eval()

    def generate(self, prompt, length=50):
        generated_text = ""
        hidden_state = None

        # tokenize the prompt
        input_seq = [Data.char_to_ix[ch] for ch in prompt]
        # tensor of dimension (N,) where N is the number of characters in the prompt
        input_seq = torch.tensor(input_seq).to(self.device)

        for i in range(length):
            # forward pass through the model
            # output is a tensor of dimension (N, vocab_size)
            output, hidden_state = self.forward(input_seq, hidden_state)

            # get a distribution over the next character
            # dist is of dimension (vocab_size,)
            probas = F.softmax(output[-1], dim=0)

            # sample a character according to the predicted distribution
            dist = Categorical(probas)
            index = dist.sample()

            generated_text += Data.ix_to_char[index.item()]

            # to continue the generation, we simply evaluate
            # the model on the last predicted character,
            # and the current hidden state
            input_seq = torch.tensor([index.item()]).to(self.device)

        return prompt + generated_text
