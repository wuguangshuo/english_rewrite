# import re
# a='which (flight) which departs'
# b=re.sub(r"(?<=\().+?(?=\))|(?<=\[).+?(?=\])", "hahah", a)
# print(b)
#
# string = "This is (a) string"
# string = a.replace("(","").replace(")","")
# print(string)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
text="plane leave from dillingham airport"
b=tokenizer.encode(text)

a = tokenizer(text, return_tensors="pt")
print(a)
print(b)
c=tokenizer.decode(b)
d=tokenizer.tokenize(text)
print(c)
print(d)

import numpy as np


sentences = []  # 存储结果的列表
tokens = text.strip().split(' ')  # 按照空格进行分词

subwords = list(map(tokenizer.tokenize, tokens))
"""
[['Nan', '##ost', '##ru', '##cture', '##d'], ['P', '##t', '-', 'alloy'], ['electro', '##cat', '##aly', '##sts'], ['for'], ['P', '##EM'], ['fuel'], ['cell'], ['oxygen'], ['reduction'], ['reaction']]
"""
subword_lengths = list(map(len, subwords))
# [5, 4, 4, 1, 2, 1, 1, 1, 1, 1]

subwords = ['[CLS]'] + [item for indices in subwords for item in indices]
# ['[CLS]', 'Nan', '##ost', '##ru', '##cture', '##d', 'P', '##t', '-', 'alloy', 'electro', '##cat', '##aly', '##sts', ...]
# cls直接加到句首，并且列表

token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
# 基于cumsum方法对长度进行累加，获取词首index，整体+1，相当于加入了cls标记占位的影响

sentences.append((tokenizer.convert_tokens_to_ids(subwords), token_start_idxs))
print('666')


# from transformers import BertTokenizerFast
# tokenizerb = BertTokenizerFast.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
# a2 = tokenizerb(text, return_tensors="pt",return_offsets_mapping=True)
# b2=tokenizerb.encode(text)
# print(a2)
# print(b2)
# c2=tokenizerb.decode(b2)
# d2=tokenizerb.tokenize(text)
# print(c2)
# print(d2)

