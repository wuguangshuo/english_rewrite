import torch
from model import TaggerRewriteModel
from transformers import BertTokenizer,AutoConfig
import numpy as np
from config import set_train_args

#加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
args = set_train_args()
config = AutoConfig.from_pretrained(args.model_path)
config.num_labels = 3
model = TaggerRewriteModel(config,args)
model.load_state_dict(torch.load('./best_model.pkl')) # 导入网络的参数
model.to(device)
print('模型加载成功')

#用户自定义输入
input_a = input("input:")
input_b = input("input:")
token_type = []

#数据处理
context_new_input = ["[CLS]"] + input_a.strip().split(' ')  + ["[SEP]"]
utterance_token =input_b.strip().split(' ') + ["[SEP]"]
new_input = context_new_input + utterance_token

ori_sentence=[]
ori_sentence.append([',', input_a, ',', input_b, ','])
subwords = list(map(tokenizer.tokenize, new_input))
subword_lengths = list(map(len, subwords))
subwords = [item for indices in subwords for item in indices]
token_start_idxs = np.cumsum([0] + subword_lengths[:-1])  # wordpiece后第一个词出现的位置
context_input_ids = tokenizer.convert_tokens_to_ids(subwords)
# 模型输入
token_type_len = context_input_ids.index(102) + 1
token_type_ids_a = [0] * token_type_len
token_type_ids_b = (len(context_input_ids) - token_type_len) * [1]
token_type_ids = token_type_ids_a + token_type_ids_b
input_mask=[1]*len(context_input_ids)
context_input_ids=torch.tensor(context_input_ids, dtype=torch.long).to(device)
input_mask=torch.tensor(input_mask, dtype=torch.long).to(device)
token_type_ids=torch.tensor(token_type_ids, dtype=torch.long).to(device)

label_starts = np.zeros(len(context_input_ids))
label_starts[[idx for idx in token_start_idxs if idx < len(context_input_ids)]] = 1
label_starts=torch.tensor(label_starts, dtype=torch.long).to(device)

context_input_ids=context_input_ids.unsqueeze(0)
input_mask=input_mask.unsqueeze(0)
token_type_ids=token_type_ids.unsqueeze(0)
label_starts=label_starts.unsqueeze(0)
start,end,insert_pos=None,None,None

outputs = model(input_ids=context_input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, token_starts=label_starts,
                start=start, end=end, insert_pos=insert_pos)
start_logits, end_logits, insert_pos_logits = outputs[1], outputs[2], outputs[3]

def find_best_answer_for_passage(start_probs, end_probs, split_index):
    best_end, best_start= find_best_answer(start_probs[1:split_index], end_probs[1:split_index])
    return best_end, best_start

def find_best_answer(start_probs, end_probs):
    start_probs, end_probs = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    prob_start, best_start = torch.max(start_probs, 1)
    prob_end, best_end = torch.max(end_probs, 1)
    if best_start <= best_end:
        return best_start, best_end
    else:
        return best_end, best_start

all_outputs=[]
context = ori_sentence[0][1].strip().split(' ')
split_index = len(context) + 1  # 加1是因为有cls影响
best_start, best_end = find_best_answer_for_passage(start_logits[0], end_logits[0], split_index)
info_pos = (best_start.cpu().numpy()[0], best_end.cpu().numpy()[0])
# 找到需要插入的单词
text = context[info_pos[0]:(info_pos[1] + 1)]
current = ori_sentence[0][3].strip().split(' ')
# 若待插入单词长度为0或者已经在回复中，明显不合理的标签预测
if len(text) == 0 or text in current:
    all_outputs.append(current)
# 找到插入位置索引，减去上文和sep影响
insert_pos = insert_pos_logits[0].argmax().cpu().numpy()
insert_pos -= (split_index + 1)
if insert_pos > 0:
    rewritten_text = current[:insert_pos] + text + current[insert_pos:]
    all_outputs.append(rewritten_text)
elif insert_pos == 0:
    rewritten_text = text + current
    all_outputs.append(rewritten_text)
else:
    all_outputs.append(current)

res=' '.join(all_outputs[0])
print(res)




