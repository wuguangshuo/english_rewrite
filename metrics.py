import os

def evaluate(a,current,pred, label,mode):
    em_count = 0.0
    if mode=='val':
        for idx,(p, l) in enumerate(zip(pred, label)):
            if p.strip()==l.strip():
                em_count+=1
        res = em_count / len(pred)
    elif mode=='test':
        if not os.path.exists('bad_case.txt'):
            os.system(r"touch {}".format('bad_case.txt'))
        output = open('bad_case.txt', 'w', encoding='utf-8')
        for idx,(data,c,p, l) in enumerate(zip(a,current,pred, label)):
            if p.strip()==l.strip():
                em_count+=1
            else:
                output.write("bad case " + str(idx) + ": \n")
                output.write("用户输入 " + str(data) + ": \n")
                output.write("当前回复: " + str(c) + "\n")
                output.write("预测回复: " + str(p) + "\n")
                output.write("真实回复: " + str(l) + "\n")
        res=em_count/len(pred)
    return res




