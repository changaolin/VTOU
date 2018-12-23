from urllib.parse import urlparse
import os
import pandas as pd
sep = "++-++"
dicurl = {'auto.sohu.com': 'qiche', 'it.sohu.com': 'hulianwang', 'health.sohu.com': 'jiankang',
                      'sports.sohu.com': 'tiyu',
                      'travel.sohu.com': 'lvyou', 'learning.sohu.com': 'jiaoyu', 'career.sohu.com': 'zhaopin',
                      'cul.sohu.com': 'wenhua',
                      'mil.news.sohu.com': 'junshi', 'house.sohu.com': 'fangchan', 'yule.sohu.com': 'yule',
                      'women.sohu.com': 'shishang',
                      'media.sohu.com': 'chuanmei', 'gongyi.sohu.com': 'gongyi', '2008.sohu.com': 'aoyun',
                      'business.sohu.com': 'shangye'}
def clean():
    """
    去除<url><contenttitle><content>的标签
    :return: 生成 urls.txt titles.txt contents.txt
    """
    with open('url.txt','r',encoding='utf-8') as u:
        new = []
        for line in u.readlines():
            line = line.replace('<url>','').replace('</url>','')
            new.append(line)
        with open('urls.txt','w',encoding='utf-8') as uw:
            uw.writelines(new)
            print(len(new))
    with open('content.txt','r',encoding='utf-8') as u:
        new = []
        for line in u.readlines():
            line = line.replace('<content>','').replace('</content>','')
            new.append(line)
        with open('contents.txt','w',encoding='utf-8') as uw:
            uw.writelines(new)
            print(len(new))

    with open('title.txt','r',encoding='utf-8') as u:
        new = []
        for line in u.readlines():
            line = line.replace('<contenttitle>','').replace('</contenttitle>','')
            new.append(line)
        with open('titles.txt','w',encoding='utf-8') as uw:
            uw.writelines(new)
            print(len(new))
def con():
    """
    用 urls.txt contents.txt titles.txt 生成dataSet
    :return: News.csv:("id,url,title,content,type")
    """
    urls = []
    titles = []
    contents = []
    type = []
    index = []
    with open('urls.txt', 'r', encoding='utf-8') as uw:
        urls = uw.readlines()
    with open('contents.txt', 'r', encoding='utf-8') as uw:
        titles = uw.readlines()
    with open('titles.txt', 'r', encoding='utf-8') as uw:
        contents = uw.readlines()
    index = [str(i) for i in range(len(urls))]

    with open('news.csv','w',encoding='utf-8') as c:
        c.write("id,url,title,content,type\n")
        all = set()
        for i in index:
            ind = int(i)
            url = urls[ind].strip('\n')
            title = titles[ind].strip('\n')
            content = contents[ind].strip('\n')
            tmp = urlparse(url)
            type = 'unknown'
            all.add(tmp.hostname)

            if tmp.hostname in dicurl.keys():
                type = dicurl[tmp.hostname]
            line = i+sep+url+sep+title+sep+content+sep+type+"\n"
            c.write(line)
    pass
def getCorpusWithTypeTmp(type='shangye'):
    with open('urls.txt', 'r', encoding='utf-8') as urlin,\
            open('contents.txt', 'r', encoding='utf-8') as contentsin,\
            open(type+'.txt','w',encoding='utf-8') as result:
        new_linses = []
        urls = urlin.readlines()
        contents = contentsin.readlines()
        i = 0
        for i in range(len(urls)):
            url = urls[i]
            tmp = urlparse(url)
            if tmp.hostname == 'business.sohu.com':
                new_linses.append(contents[i])
        result.write('\n'.join(new_linses))

    pass
def getCorpusWithType(type='shangye'):
    """
    得到特定类别的数据
    :param type:con.dicurl.values()
    :return:
    """
    data = pd.read_csv('news.csv',sep=sep,error_bad_lines=False,engine='python',encoding='utf-8')
    mydata = data[data['type']==type]
    mydata.to_csv(type+'.csv',sep=sep,header=False)
    with open(type+".txt",'w',encoding='utf-8') as writeto:
        values = mydata['content'].value_counts()
        writeto.write('\n'.join(values))
    print(mydata.shape)
    print(mydata.head())

def splitCorpus():
    """
    转换并分割搜狗语料库
    :return: 生成三个文件：url.txt,title.txt,content.txt
    """
    os.system("./split.sh")


if __name__ == '__main__':
    # splitCorpus()
    # clean()
    # con()
    getCorpusWithType()
    # getCorpusWithTypeTmp()
    pass