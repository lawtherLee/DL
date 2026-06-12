import jieba
import torch
import torch.nn as nn

text = "欢迎来到黑马程序员学习人工智能相关课程"

words = jieba.lcut(text)
# print(words)

unique_words = list(set(words))
# print(unique_words)

word_to_idx = {word: idx for idx, word in enumerate(unique_words)}

word_indices = [word_to_idx[word] for word in words]
print(word_indices)

# print(word_to_idx)

input_tensor = torch.tensor(word_indices)
print(input_tensor)

# 7. 创建词嵌入层
embedding = nn.Embedding(num_embeddings=len(unique_words), embedding_dim=4)

# 8. 得到词向量
embedded = embedding(input_tensor)

print(embedded)
print(embedded.shape)
