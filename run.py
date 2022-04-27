from transformers import AutoConfig
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
from sklearn.utils import shuffle

from model import TaggerRewriteModel
from utils import seed_everything,set_logger
from config import set_train_args
from data_utils import TaggerRewriterDataset
from decode import validate

#超参数设置
args = set_train_args()
seed_everything(args.seed)
#记录tensorboard和日志
writer = SummaryWriter(os.path.join(args.tensorboard_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
set_logger(args.log_dir)
device='cuda' if torch.cuda.is_available() else 'cpu'
#读取和划分数据集 正样本label_type为0,代表需要改写,负样本label_type为1，代表需要改写
df = pd.read_csv('./dataset/new_data.csv', sep=",", names=['a', 'b', 'current', 'label','replace_pos','label_type'], dtype=str,
                   encoding='utf-8')[1:]
df=shuffle(df)
#去除数据集中的(),保留其中的内容
# df['label'] = df['label'].apply(lambda x: x.replace("(","").replace(")",""))
df.dropna(how='any', inplace=True)

#划分数据集
train_length = int(len(df) * 0.8)
train_df = df.iloc[:train_length]
valid_df_data = df.iloc[train_length:]

#划分验证集和测试集
test_length = int(len(valid_df_data) * 0.5)
valid_df=valid_df_data.iloc[:test_length]
test_df=valid_df_data.iloc[test_length:]

#训练集处理
train_set = TaggerRewriterDataset(train_df,args)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,collate_fn=train_set.tagger_collate_fn)

valid_set = TaggerRewriterDataset(valid_df,args,valid=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size,shuffle=False, collate_fn=valid_set.tagger_collate_fn)

test_set = TaggerRewriterDataset(test_df,args,valid=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size,shuffle=False, collate_fn=test_set.tagger_collate_fn)

#模型设置
config = AutoConfig.from_pretrained(args.model_path)
config.num_labels = 3
model = TaggerRewriteModel(config,args)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
param_optimizer = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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
logging.info("--------Start Training!--------")
for epoch in range(1, args.epoch):
    model.train()
    losses=0.0#每一轮都需要将losses重新设置为0
    for i, (ori_sen, token, token_type,token_starts ,start, end, insert_pos) in enumerate(tqdm(train_loader)):
        input_mask = (token > 0).to(device)
        token, input_mask, token_type,token_starts , start, end, insert_pos = \
            token.to(device), input_mask.to(device), token_type.to(device),token_starts.to(device), start.to(
                device), end.to(device), insert_pos.to(device)
        outputs = model(input_ids=token, attention_mask=input_mask, token_type_ids=token_type,token_starts=token_starts,
                        start=start, end=end, insert_pos=insert_pos)
        loss = outputs[0]
        losses+=loss.item()
        optimizer.zero_grad()
        # model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss=losses/len(train_loader)
    logging.info('Epoch: {},train loss: {}'.format(epoch, train_loss))
    writer.add_scalar('Training/training loss', train_loss, epoch)## tensorboard --logdir "./runs"启动

    valid_metrics = validate(model, valid_loader, valid_df, args,mode='val')
    logging.info('Epoch: {},vailm em: {}'.format(epoch, valid_metrics))
    current_score = valid_metrics
    if current_score > best_valid_score:
        print("Epoch: {}".format(epoch)+'保存模型')
        torch.save(model.state_dict(),'./best_model.pkl')

print("测试集预测")
model.load_state_dict(torch.load('./best_model.pkl'))
print('模型加载完成')
test_score=validate(model, valid_loader, valid_df, args,mode='test')
print(test_score)

