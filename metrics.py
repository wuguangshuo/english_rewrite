
def evaluate(pred, label):
    em_count = 0.0
    for p, l in zip(pred, label):
        if p==l:
            em_count+=1
    res=em_count/len(pred)
    return res