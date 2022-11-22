import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PREFIX = "Florian's password is "


class Vocab:
    """
    The vocabulary of our language model.
    """

    # The language model emits text character by character. The vocabulary is the set of all possible characters.
    chars = ['\n', ' ', '!', '"', '&', "'", '(', ')', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8',
             '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
             'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '¢', '£', '¨', '©',
             'ª', '®', '°', '±', '´', '¶', '»', '¼', '½', 'Â', 'Ã', 'â', 'Ÿ', '€', '™']
    size = len(chars)

    # utilities to convert between characters and "tokens" (indices into the vocabulary)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}


class RNN(nn.Module):
    """
    A simple recurrent neural network (RNN) language model.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_seq, hidden_state):
        """
        Forward pass of the RNN.
        :param input_seq: the tokenized input sequence of length seq_len
        :param hidden_state: the model's current state
        :return: the model's output, of dimension (seq_len, Vocab.size), and the new state
        """
        embedding = self.embedding(torch.unsqueeze(input_seq, dim=-1))
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output.squeeze(1), (hidden_state[0].detach(), hidden_state[1].detach())


def load_lm(ckpt_path="infoseclab/data/secret_model.pth", device="cuda"):
    """
    Load the language model from the checkpoint.
    :param ckpt_path: the pretrained model checkpoint
    :param device: the device to load the model on
    :return: the language model
    """
    rnn = RNN(Vocab.size, Vocab.size, 512, 3)
    rnn.load_state_dict(torch.load(ckpt_path, map_location=device))
    rnn.to(device)
    rnn.eval()
    rnn.device = device
    return rnn


def get_loss(lm, seq):
    """
    Compute the loss of the language model on a string of characters.
    :param lm: the language model
    :param seq: the string of characters
    :return: the loss
    """

    # tokenize the sequence
    seq = torch.tensor([Vocab.char_to_ix[ch] for ch in seq], device=lm.device)

    # feed the n-1 first characters to the model.
    # the model will output a distribution for each of the [2nd, 3rd, ..., n-th] characters
    output, _ = lm.forward(seq[:-1], None)

    # apply the cross-entropy loss for each predicted character in the sequence and average the losses
    loss = lm.loss_fn(output, seq[1:])
    return loss


def generate(lm, prompt, length=50):
    """
    Generate a sequence of characters by sampling from the language model.
    :param lm: the language model
    :param prompt: the prompt to start the generation from
    :param length: the number of characters to generate
    :return: the prompt concatenated with the generated sequence
    """
    generated_text = ""
    hidden_state = None

    # tokenize the prompt
    input_seq = [Vocab.char_to_ix[ch] for ch in prompt]
    # tensor of dimension (N,) where N is the number of characters in the prompt
    input_seq = torch.tensor(input_seq).to(lm.device)

    for i in range(length):
        # forward pass through the model
        # output is a tensor of dimension (N, vocab_size)
        output, hidden_state = lm.forward(input_seq, hidden_state)

        # get a distribution over the next character
        # probas is of dimension (vocab_size,)
        probas = F.softmax(output[-1], dim=0)

        # sample a character according to the predicted distribution
        dist = Categorical(probas)
        index = dist.sample()
        generated_text += Vocab.ix_to_char[index.item()]

        # to continue the generation, we simply evaluate
        # the model on the last predicted character,
        # and the current state
        input_seq = torch.tensor([index.item()]).to(lm.device)

    return prompt + generated_text
