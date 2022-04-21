from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import numpy as np

class TaggerRewriterDataset(Dataset):
    def __init__(self, df,args,valid=False):
        self.a = df['a'].values.tolist()
        self.is_valid = valid#判断是否信息抽取
        self.current = df['current'].values.tolist()
        self.label = df['label'].values.tolist()
        self.pointer = []#储存指针信息
        self.valid_label = []#记录label
        self.ori_sentence=[]#记录 文字信息
        self.context_new_input=[]#记录模型输入
        self.utterance_token = []#记录模型输入
        self.new_input=[]#记录模型输入
        self.generate_label()#生成数据
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path)

    def generate_label(self):
        drop_item = 0
        for i in range(len(self.a)):
            #无论训练集测试集都需要这一步
            context_new_input = ["[CLS]"] +self.a[i].strip().split(' ')+["[SEP]"]
            utterance_token = self.current[i].strip().split(' ')+["[SEP]"]
            new_input=context_new_input+utterance_token
            # 改写或者作为验证集时不对关键信息进行抽取
            if self.is_valid:
                _label = [None] * 3
                self.pointer.append(_label)
                self.ori_sentence.append([',', self.a[i], ',', self.current[i], ','])
                self.valid_label.append(self.label[i])
                self.context_new_input.append(context_new_input)
                self.utterance_token.append(utterance_token)
                self.new_input.append(new_input)
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
            text_len=len(text)#判断插入单词或者短语
            tmp_a = self.a[i].strip().split(' ')
            #插入单词
            if text_len==1:
                text=' '.join(x for x in text)
                # 获取插入文本及位置
                if text in tmp_a:
                    start = tmp_a.index(text)+1#加1是为了消除cls影响
                    end = start + text_len - 1
                elif text in tmp_current:
                    start = tmp_current.index(text) + len(tmp_a) + 2#加2是为了消除cls和sep影响
                    end = start + len(text) - 1
                else:
                    drop_item += 1#增益信息不在原句中
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
            self.ori_sentence.append([',', self.a[i], ',', self.current[i], ','])
            self.valid_label.append(self.label[i])
            self.context_new_input.append(context_new_input)
            self.utterance_token.append(utterance_token)
            self.new_input.append(new_input)
        print('数据总数 ', len(self.context_new_input), '丢弃样本数目 ', drop_item)

    def __len__(self):
        return len(self.context_new_input)

    def __getitem__(self, idx):
        return  self.ori_sentence[idx],\
                self.context_new_input[idx], \
                self.utterance_token[idx], \
                self.new_input[idx], \
                self.pointer[idx][0],\
                self.pointer[idx][1],\
                self.pointer[idx][2]

    def tagger_collate_fn(self,batch):
        # 关键数据处理部分
        ori_sen, context_input,utterance_input,new_input,start, end,insert_pos = zip(*batch)
        context_input_ids_list=[]#记录模型输入
        token_type_ids_list=[]#记录输入类型
        token_start_idxs_list=[]#记录那些词需要取出，防止logits无法对齐
        for texta,textb in (zip(context_input,utterance_input)):
            data=texta+textb
            subwords = list(map(self.tokenizer.tokenize,data))
            subword_lengths = list(map(len, subwords))
            subwords = [item for indices in subwords for item in indices]
            token_start_idxs = np.cumsum([0] + subword_lengths[:-1])#wordpiece后第一个词出现的位置
            context_input_ids=self.tokenizer.convert_tokens_to_ids(subwords)
            #toke_type输入
            token_type_len=context_input_ids.index(102)+1
            token_type_ids_a=[0]*token_type_len
            token_type_ids_b=(len(context_input_ids)-token_type_len)*[1]
            token_type_ids=token_type_ids_a+token_type_ids_b

            token_start_idxs_list.append(token_start_idxs)
            context_input_ids_list.append(context_input_ids)
            token_type_ids_list.append(token_type_ids)

        #填充至同一长度
        batch_len=len(ori_sen)
        max_len=max([len(s) for s in context_input_ids_list])
        batch_data = 0 * np.ones((batch_len, max_len))
        batch_token_type = 0 * np.ones((batch_len, max_len))
        batch_label_starts=[]

        for j in range(batch_len):
            cur_len=len(context_input_ids_list[j])
            batch_data[j][:cur_len]=context_input_ids_list[j]

        for j in range(batch_len):
            cur_len=len(token_type_ids_list[j])
            batch_token_type[j][:cur_len]=token_type_ids_list[j]

        #方便后面取出去除word piece对模型的影响
        for j in range(batch_len):
            label_start_idx = token_start_idxs_list[j]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
        batch_data=torch.tensor(batch_data,dtype=torch.long)
        batch_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)

        if start[0] is not None:
            start = torch.tensor(start)
            end = torch.tensor(end)
            insert_pos = torch.tensor(insert_pos)
        return ori_sen, batch_data,batch_token_type,batch_label_starts ,start, end, insert_pos