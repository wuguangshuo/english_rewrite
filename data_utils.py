from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import torch

class TaggerRewriterDataset(Dataset):
    def __init__(self, df, tokenizer, valid=False):
        self.a = df['a'].values.tolist()
        self.is_valid = valid
        self.current = df['current'].values.tolist()
        self.label = df['label'].values.tolist()
        self._tokenizer = tokenizer
        self.ori_sentence = []
        self.sentence = []
        self.token_type = []
        self.pointer = []
        self.context_len = []
        self.valid_index = []
        self.valid_label = []
        self.label_type = []
        self.generate_label()

    def tokenize_english(self,sen):
        temp = []
        for word in sen.strip().split(' '):
            if word in self._tokenizer.vocab:
                temp.append(word)
            else:
                temp.append("[UNK]")
        return temp

    def generate_label(self):
        # 全部采用指针抽取
        # 根据改写的数据对原始数据进行标注
        # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
        # 确实江西炒粉要用瓦罐汤 特产 没错是我老家的特产 没错江西炒粉是我老家的特产
        # 为什么讨厌张艺兴       我喜欢张艺兴 很可爱啊       我也喜欢他     我也喜欢张艺兴

        # start,end,insert,start_ner,end_ner
        drop_item = 0
        for i in range(len(self.a)):
            new_token_type = []
            context_new_input = ["[CLS]"] +self.tokenize_english(self.a[i])+["[SEP]"]
            new_token_type.extend([0] * len(context_new_input))
            utterance_token = self.tokenize_english(self.current[i])+["[SEP]"]
            new_input = context_new_input + utterance_token
            new_token_type.extend([1]*len(utterance_token))

            # 改写或者作为验证集时不对关键信息进行抽取
            if self.is_valid:
                # 改写的负样本
                _label = [None] * 3
                self.pointer.append(_label)
                self.sentence.append(self._tokenizer.convert_tokens_to_ids(new_input))
                self.token_type.append(new_token_type)
                self.context_len.append(context_new_input)
                self.ori_sentence.append([',', self.a[i], ',', self.current[i], ','])
                self.valid_index.append(i)
                self.valid_label.append(self.label[i])
                continue
            # -----寻找增加的信息------------------
            text_start, text_end = 0, 0
            tmp_current = self.current[i].strip().split(' ')
            tmp_label = self.label[i].strip().split(' ')
            for j in range(len(tmp_label)):
                if j >= len(tmp_current):
                    text_start = j
                    break
                if tmp_current[j] == tmp_label[j]:
                    continue
                else:
                    text_start = j
                    break
            for j in range(len(tmp_label)):
                if j >= len(tmp_current):
                    text_end = j
                    break
                if tmp_current[::-1][j] == tmp_label[::-1][j]:
                    continue
                else:
                    text_end = j
                    break
            text = tmp_label[text_start:len(tmp_label) - text_end]
            text_len=len(text)
            tmp_a = self.a[i].strip().split()
            #插入单词
            if text_len==1:
                text=' '.join(x for x in text)
                # 获取插入文本及位置
                if text in tmp_a:
                    start = tmp_a.index(text) + 1
                    end = start + text_len - 1
                elif text in tmp_current:
                    start = tmp_current.index(text) + len(tmp_a) + 2
                    end = start + len(text) - 1
                else:
                    drop_item += 1
                    continue
            #插入短语
            else:
                text=text[:1]
                start_text=' '.join(x for x in text)
                # 获取插入文本及位置
                if start_text in tmp_a:
                    start = tmp_a.index(start_text) + 1
                    end = start + text_len - 1
                elif start_text in tmp_current:
                    start = tmp_current.index(start_text) + len(tmp_a) + 2
                    end = start + len(start_text) - 1
                else:
                    drop_item += 1#对比了一下,drop的都是不需要改写的文本（因为改写和不该写的回复一样)  在这里发现了数据集的一些错误，可参考第5722条
                    continue
            # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
            insert_pos = len(tmp_current)-text_end + len(context_new_input)
            self.pointer.append([start,end,insert_pos])
            self.sentence.append(self._tokenizer.convert_tokens_to_ids(new_input))
            self.token_type.append(new_token_type)
            self.context_len.append(context_new_input)
            self.ori_sentence.append([',', self.a[i], ',', self.current[i], ','])
            self.valid_label.append(self.label[i])
            self.valid_index.append(i)
        print('数据总数 ', len(self.sentence), '丢弃样本数目 ', drop_item)


    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return  self.ori_sentence[idx],\
                torch.LongTensor(self.sentence[idx]),  \
                torch.LongTensor(self.token_type[idx]),\
                self.pointer[idx][0],\
                self.pointer[idx][1],\
                self.pointer[idx][2]


def tagger_collate_fn(batch):
    # start, end, insert_pos, start_ner, end_ner = 0,0,0,0,0
    ori_sen, token, token_type, start, end,insert_pos = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    if start[0] is not None:
        start = torch.tensor(start)
        end = torch.tensor(end)
        insert_pos = torch.tensor(insert_pos)
    return ori_sen, token, token_type, start, end, insert_pos