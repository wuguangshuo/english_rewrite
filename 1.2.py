import pandas as pd
df = pd.read_csv('./dataset/df_rewrite_airline_music_and_contact_0401.csv', sep=",", names=['a', 'b', 'current', 'label','replace_pos'], dtype=str,
                   encoding='utf-8')[1:]

df['label'] = df['label'].apply(lambda x: x.replace("(","").replace(")",""))
res=0
for i in range(len(df)):
    tmp_current = df.current[i].strip().split(' ')
    tmp_label = df.label[i].strip().split(' ')
    # 如果原始语句所有词汇都在改写中，则改写为插入新语句
    for word in tmp_current:
        if word not in tmp_label:
            res+=1
print(res)