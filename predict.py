import torch
import config
from model import TaggerRewriteModel
from transformers import BertTokenizer,BertConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
vocab=tokenizer.vocab

def tokenize_english(sen):
    temp = []
    for word in sen.strip().split(' '):
        if word in vocab:
            temp.append(word)
        else:
            temp.append("[UNK]")
    return temp

text_a = input("input:")
text_b = input("input:")
token_type = []

text_a= ["[CLS]"] +tokenize_english(text_a)+["[SEP]"]
text_b=tokenize_english(text_b)+["[SEP]"]
new_input = text_a + text_b

token_type.extend([0] * len(text_a))
token_type.extend([1] * len(text_b))

sentence=tokenizer.convert_tokens_to_ids(new_input)
model = torch.load('./best_model.pkl')
print('模型加载成功')

input_mask = (sentence > 0).to(device)
token, input_mask, token_type, start, end, insert_pos = \
    token.to(device), input_mask.to(device), token_type.to(device), start.to(
        device), end.to(device), insert_pos.to(device)




