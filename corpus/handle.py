sep=' '
flags = r'[。！？]'
import re,os
line_max = 20
max_len = 60
def word2tag(ori,base='.'):
    ori = os.path.join(base,ori)
    tmp = os.path.join(base,"tmp.txt")
    with open(ori,'r',encoding='utf-8') as input_data ,open(tmp,'w',encoding='utf-8') as output_data:
        for line in input_data.readlines():
            line = line.strip()
            i = 0
            while i < len(line):
                if line[i] == '{':
                    i+=2
                    temp = ""
                    while line[i]!="}":
                        temp+=line[i]
                        i+=1
                    i+=2
                    word=temp.split(":")
                    sen=word[1]
                    output_data.write(sen[0]+"/B_"+word[0]+sep)
                    for j in sen[1:len(sen)-1]:
                        output_data.write(j+"/M_"+word[0]+sep)
                    output_data.write(sen[-1]+"/E_"+word[0]+sep)
                else:
                    output_data.write(line[i]+"/O"+sep)
                    i+=1

def splitText(outp,base='.'):
    outp = os.path.join(base, outp)
    tmp = os.path.join(base, "tmp.txt")
    with open(tmp,'r',encoding='utf-8') as readin,\
        open(outp,'w',encoding='utf-8') as writeto:
        for line in readin.readlines():
            if len(line.strip()) == 0:
                continue
            lines = re.split(flags,line)
            for l in lines:
                if len(l) > line_max:
                    ll = l.split("，")
                    writeto.write("\n".join(ll))
                else:
                    writeto.write("\n"+l)
        pass
def preText(ori,outp,base):
    word2tag(ori,base)
    splitText(outp,base)