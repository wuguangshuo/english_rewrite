import torch

from tqdm import tqdm
import numpy as np

from metrics import evaluate

def find_best_answer(start_probs, end_probs):
    start_probs, end_probs = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    prob_start, best_start = torch.max(start_probs, 1)
    prob_end, best_end = torch.max(end_probs, 1)
    if best_start <= best_end:
        return best_start, best_end
    else:
        return best_end, best_start

def find_best_answer_for_passage(start_probs, end_probs, split_index):
    best_end, best_start= find_best_answer(start_probs[1:split_index], end_probs[1:split_index])
    return best_end, best_start


def predict(model, valid_loader):
    model.eval()
    all_outputs = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for i, (ori_sen, token, token_type, start, end, insert_pos) in enumerate(tqdm(valid_loader)):
            input_mask = (token > 0).to(device)
            token, input_mask, token_type = \
                token.to(device), input_mask.to(device), token_type.to(device)
            outputs = model(input_ids=token, attention_mask=input_mask, token_type_ids=token_type,
                            start=None, end=end, insert_pos=insert_pos)
            start_logits, end_logits, insert_pos_logits = outputs[1], outputs[2], outputs[3]
            # 解码出真实label
            for i in range(len(token)):
                context=ori_sen[i][1].strip().split(' ')
                split_index = len(context)+1
                best_start, best_end= find_best_answer_for_passage(start_logits[i], end_logits[i], split_index)
                info_pos = (best_start.cpu().numpy()[0], best_end.cpu().numpy()[0])
                text = context[info_pos[0]:(info_pos[1]+1)]
                context_len = sum(token_type[i].cpu().numpy() == 0)
                current=ori_sen[i][3].strip().split()
                if len(text) == 0 or text in current:
                    all_outputs.append(current)
                    continue
                # insert_pos = insert_pos_logits[i].argmax().cpu().numpy()
                insert_pos=insert_pos_logits[i].cpu().numpy()
                insert_pos = np.argmax(insert_pos)
                insert_pos=insert_pos-context_len
                if insert_pos>0:
                    rewritten_text = current[:insert_pos]+text+current[insert_pos:]
                    all_outputs.append(rewritten_text)
                    continue
                elif insert_pos==0:
                    rewritten_text = text+current
                    all_outputs.append(rewritten_text)
                    continue
                else:
                    all_outputs.append(current)
                    continue
    return all_outputs


def validate(model, valid_loader, valid_df, args,mode='val'):
    predictions = predict(model, valid_loader)
    valid_label = valid_df['label'].tolist()
    a = valid_df['a'].tolist()
    b = valid_df['b'].tolist()
    current = valid_df['current'].tolist()
    # print(len(predictions),len(valid_label))
    predictions = [' '.join(x) for x in predictions]
    valid_metric = evaluate(a,current,predictions, valid_label,mode)
    print(valid_metric)
    print('------------')
    if mode=='val':
        for i, (a, b, current, p, l) in enumerate(zip(a, b, current, predictions, valid_label)):
            print(a,' | ', b,' | ', current,' | ', p,' | ', l)
            if i >= args.print_num:
                break
    return valid_metric