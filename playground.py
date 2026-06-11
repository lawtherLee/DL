import torch
import jieba
from imageio.plugins import lytro
from plotly.graph_objs.layout import shape
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time


def build_vocab():
    unique_words, all_words = [], []
    for line in open("./data/jaychou_lyrics.txt", "r", encoding="utf-8"):
        words = jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    word_count = len(unique_words)
    word_to_idx = {word: i for i, word in enumerate(unique_words)}
    corpus_idx = []
    for words in all_words:
        tem = []
        for word in words:
            tem.append(word_to_idx[word])
        tem.append(word_to_idx[" "])
        corpus_idx.extend(tem)

    return unique_words, word_to_idx, word_count, corpus_idx


class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.word_count = len(self.corpus_idx)
        self.number = self.word_count // self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        x = torch.tensor(self.corpus_idx[start:end])
        y = torch.tensor(self.corpus_idx[start + 1 : end + 1])
        return x, y


class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):
        super().__init__()
        self.ebd = nn.Embedding(unique_word_count, 128)
        self.rnn = nn.RNN(128, 256, 1)
        self.output = nn.Linear(256, unique_word_count)

    def forward(self, inputs, hidden):
        embd = self.ebd(inputs)
        output, hidden = self.rnn(embd.transpose(0, 1), hidden)
        output = self.output(output.reshape(shape=(-1, output.shape[-1])))
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(1, bs, 256)


def train(lyrics_dataloader, model, criterion, optimizer):
    start, iter_num, total_loss = time.time(), 0, 0
    for epoch in range(epochs):
        for x, y in lyrics_dataloader:

            init_hidden = model.init_hidden(5)
            output, hidden = model(x, init_hidden)
            y = torch.transpose(y, 0, 1).reshape(-1)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iter_num += 1


if __name__ == "__main__":
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # print(f"词的数量: {unique_word_count}")
    # print(f"去重后的词: {unique_words}")
    # print(f"每个词的索引: {word_to_index}")
    # print(f"文档中每个词对应的索引: {corpus_idx}")
    lyrics = LyricsDataset(corpus_idx, 32)
    epochs = 10
    model = TextGenerator(unique_word_count)
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
