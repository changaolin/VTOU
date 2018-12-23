import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from utils import taLogging
logger = taLogging.getFileLogger(name='util',file='log/util.log')
sep=' '
max_len = 60
flags = r'[。！？；]'
import re
line_max = 20
def get_entity(x,y,id2tag):
    """
    组合实体
    :param x: text
    :param y: pre
    :param id2tag:
    :return:
    """
    entity=""
    res=[]
    for i in range(len(x)): #for every sen
        for j in range(len(x[0])): #for every word
            if y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity=id2tag[y[i][j]][2:]+':'+x[i][j]
            elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 :
                entity+=x[i][j]
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 :
                entity+=x[i][j]
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res
def padding(ids):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        ids.extend([0]*(max_len-len(ids)))
        return ids

def padding_word(sen):
    if len(sen) >= max_len:
        return sen[:max_len]
    else:
        return sen

def test_input(model,sess,word2id,id2tag,batch_size):
    while True:
        text = input("Enter your input: ")
        text = re.split(flags, text)
        text_id=[]
        for sen in text:
            word_id=[]
            for word in sen:
                if word in word2id:
                    word_id.append(word2id[word])
                else:
                    word_id.append(word2id["unknow"])
            text_id.append(padding(word_id))
        zero_padding=[]
        zero_padding.extend([0]*max_len)
        text_id.extend([zero_padding]*(batch_size-len(text_id)))
        feed_dict = {model.input_data:text_id}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        entity = get_entity(text,pre[0],id2tag)
        print( 'result:')
        for i in entity:
            print (i)
    pass
def write_entity(outp,x,y,id2tag):
    '''
    注意，这个函数每次使用是在文档的最后添加新信息。
    '''
    entity=''
    for i in range(len(x)):
        if y[i]==0:
            continue
        if id2tag[y[i]][0]=='B':
            entity=id2tag[y[i]][2:]+':'+x[i]
        elif id2tag[y[i]][0]=='M' and len(entity)!=0:
            entity+=x[i]
        elif id2tag[y[i]][0]=='E' and len(entity)!=0:
            entity+=x[i]
            print(entity)
            outp.write(entity+' ')
            entity=''
        else:
            entity=''
    return

def write_entity_for_tag(x,y,id2tag):
    '''
    注意，这个函数每次使用是在文档的最后添加新信息。
    '''
    entity=''
    line = []
    tag = ''
    le = min(len(x),max_len)
    for i in range(le):
        if y[i]==0:
            continue
        if id2tag[y[i]][0]=='B':
            tag = id2tag[y[i]][2:]
            line.append(x[i]+"/"+"B_"+tag)
            entity=id2tag[y[i]][2:]+':'+x[i]
        elif id2tag[y[i]][0]=='M' and len(entity)!=0:
            line.append(x[i] + "/" + "M_" + tag)
            entity+=x[i]
        elif id2tag[y[i]][0]=='E' and len(entity)!=0:
            line.append(x[i] + "/" + "E_" + tag)
            entity+=x[i]
            entity=''
        elif id2tag[y[i]][0]=='O':
            line.append(x[i] + "/" + "O")
            entity = ''
            pass
        else:
            entity = ''
    l = sep.join(line)
    return l
def splitText(inp,oup):
    new_lines = []
    oup=oup+".tmp"
    logger.debug("start split Text")
    with open(inp,'r',encoding='utf-8') as readin,\
        open(oup,'w',encoding='utf-8') as writeto:
        i = 0
        for line in readin.readlines():
            if len(line.strip()) == 0:
                continue
            i+=1
            if (i % 100) == 0:
                i = 1
                logger.debug(".")
            if len(line.strip()) == 0:
                continue
            lines = re.split(flags,line)
            for l in lines:
                if len(l) > line_max:
                    ll = l.split("，")
                    writeto.write("\n".join(ll))
                    new_lines.extend(ll)
                else:
                    writeto.write(l)
                    new_lines.extend([l])
    logger.debug("End split Text")
    return new_lines
def tagText(input_path, output_path, model, sess, word2id, id2tag, batch_size,pre=False):
    info = {
        'input':input_path,
        "output":output_path
    }
    logger.debug(info)
    if pre == False:
        lines = splitText(input_path,output_path)
    else:
        oup = output_path + ".tmp"
        with open(oup,'r',encoding='utf-8') as readin:
            lines = readin.readlines()
    text_id = []
    text = []
    y = 0
    for line in lines:
        y += 1
        if (y % 1000) == 0:
            y = 1
            logger.debug("read Text+..")
        if len(line.strip()) == 0:
            continue
        word_id = []
        for word in line:
            if word in word2id:
                word_id.append(word2id[word])
            else:
                word_id.append(word2id["unknow"])
        text_id.append(padding(word_id))
        text.append(padding_word(line))
    zero_padding = []
    zero_padding.extend([0]*max_len)
    text_id.extend([zero_padding]*(batch_size - len(text_id)%batch_size))
    text_id = np.asarray(text_id)
    text_id = text_id.reshape(-1,batch_size,max_len)
    predict = []
    logger.debug("len(text_id):"+str(len(text_id)))
    for index in range(len(text_id)):
        if (index % 1000) == 0:
            logger.debug("pre:"+str(index))
        feed_dict = {model.input_data: text_id[index]}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        predict.append(pre[0])
    predict = np.asarray(predict).reshape(-1, max_len)
    with open(output_path,'w',encoding='utf-8') as outp:
        logger.debug("len(text):"+str(len(text)))
        for index in range(len(text)):
            if (index % 1000) == 0:
                logger.debug("get entity:" + str(index))
            result = write_entity_for_tag(text[index], predict[index], id2tag)
            outp.write(result+'\n')
    pass

def savePkl(inpf,oupf,base):
    datas = []
    labels = []
    tags = set()
    inpf = os.path.join(base,inpf)
    oupf = os.path.join(base, oupf)
    logger.debug("start save PKl")
    with open(inpf,'r',encoding='utf-8') as inp:
        y = 0
        for line in inp.readlines():
            y += 1
            if (y % 1000) == 0:
                y = 1
                logger.debug("*")
            line = line.split()
            linedata = []
            linelabel = []
            numNotO = 0
            for word in line:
                word = word.split('/')
                linedata.append(word[0])
                linelabel.append(word[1])
                tags.add(word[1])
                if word[1] != 'O':
                    numNotO += 1
            if numNotO != 0:
                datas.append(linedata)
                labels.append(linelabel)
    logger.debug(len(datas))
    logger.debug(tags)
    logger.debug(len(labels))
    new_datas = []
    for data in datas:
        new_datas.extend(data)
    allwords = pd.Series(new_datas).value_counts()
    set_words = allwords.index

    set_ids = range(1,len(set_words)+1)

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids,index=set_words)
    id2word = pd.Series(set_words,index=set_ids)
    tag2id = pd.Series(tag_ids,index=tags)
    id2tag = pd.Series(tags,index=tag_ids)

    word2id["unknow"] = len(word2id)+1
    def padding(type,words):
        if type == "X":
            ids = list(word2id[words])
        if type == "Y":
            ids = list(tag2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0]*(max_len - len(ids)))
        return ids

    df_datas = pd.DataFrame({"words":datas,"tags":labels},index=list(range(len(datas))))
    df_datas['x'] = df_datas["words"].apply(lambda x:padding("X", x))
    df_datas['y'] = df_datas["tags"].apply(lambda x:padding("Y", x))
    x = np.asarray(list(df_datas['x'].values))
    y = np.asarray(list(df_datas['y'].values))
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12)
    def dumpPkl(ll,out):
        for l in ll:
            pickle.dump(l,out)
    with open(oupf,'wb') as out:
        x_train = x
        y_train = y
        ll = [word2id,id2word,tag2id,id2tag,x_train,y_train,x_test,y_test]
        dumpPkl(ll,out)
        pass
    logger.debug("End save PKl")

