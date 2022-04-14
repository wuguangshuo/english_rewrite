from transformers import BertTokenizer,BertConfig
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers.optimization import get_cosine_schedule_with_warmup,AdamW
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from tqdm import tqdm
import os
import time
import logging

from model import TaggerRewriteModel
from utils import seed_everything,set_logger
from config import set_train_args
from data_utils import TaggerRewriterDataset,tagger_collate_fn
from decode import validate

#超参数设置
args = set_train_args()
seed_everything(args.seed)
#记录tensorboard和日志
writer = SummaryWriter(os.path.join(args.tensorboard_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
set_logger(args.log_dir)
device='cuda' if torch.cuda.is_available() else 'cpu'
#读取和划分数据集
df = pd.read_csv('./dataset/df_rewrite_airline_music_and_contact_0401.csv', sep=",", names=['a', 'b', 'current', 'label','replace_pos'], dtype=str,
                   encoding='utf-8')[1:]
df['label'] = df['label'].apply(lambda x: x.replace("(","").replace(")",""))
df.dropna(how='any', inplace=True)
train_length = int(len(df) * 0.8)
train_df = df.iloc[:train_length].iloc[:, :]
valid_df = df.iloc[train_length:]
#训练集处理
tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
train_set = TaggerRewriterDataset(train_df, tokenizer)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=tagger_collate_fn)

valid_set = TaggerRewriterDataset(valid_df, tokenizer, valid=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size,shuffle=False, collate_fn=tagger_collate_fn)
#模型设置
config = BertConfig("google/bert_uncased_L-4_H-256_A-4")
config.num_labels = 3
model = TaggerRewriteModel(config)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=args.lr, eps=args.adam_epsilon)
t_total = int(len(train_loader))* args.epoch
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0.1 * t_total,
                                            num_training_steps=t_total)
#模型训练
criterion = CrossEntropyLoss().cuda()
best_valid_score=0.0
losses = 0.0
logging.info("--------Start Training!--------")
for epoch in range(1, args.epoch):
    model.train()
    for i, (ori_sen, token, token_type, start, end, insert_pos) in enumerate(tqdm(train_loader)):
        input_mask = (token > 0).to(device)
        token, input_mask, token_type, start, end, insert_pos = \
            token.to(device), input_mask.to(device), token_type.to(device), start.to(
                device), end.to(device), insert_pos.to(device)
        outputs = model(input_ids=token, attention_mask=input_mask, token_type_ids=token_type,
                        start=start, end=end, insert_pos=insert_pos)
        loss = outputs[0]
        losses+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss=losses/len(train_loader)
    logging.info('Epoch: {},train loss: {}'.format(epoch, train_loss))
    writer.add_scalar('Training/training loss', train_loss, epoch)## tensorboard --logdir "./runs"启动

    valid_metrics = validate(model, valid_loader, valid_df, args)
    logging.info('Epoch: {},vailm em: {}'.format(epoch, valid_metrics))
    current_score = valid_metrics
    if current_score > best_valid_score:
        print("Epoch: {}".format(epoch)+'保存模型')
        torch.save(model.state_dict(),'./best_model.pkl')

