# VTOU
实验内容：  

1.处理标注语料：[BosonNLP_NER_6C.txt](https://bosonnlp.com/resources/BosonNLP_NER_6C.zip)    
    
2.预训练实体标注模型  

3.对搜狗实验室数据处理(News文件夹)  

    1.下载数据  
    
    2.执行split脚本分割数据  
    
    3.执行handleNews.py 生成数据集，本实验抽取出其中商业部分（shangye.txt）  
    
    4.将shangye.txt移动到corpus目录下，方便数据处理  
    

4.命令介绍
     `python main.py prepare` :处理boson语料，保存语料pkl,训练模型，用于标注搜狗语料  
    
     `python main.py tag`    :标注 搜狗语料，生成shangye.pkl  
    
     `python train `  :使用预处理的boson语料进行训练  
    
     `python main.py train taged` :使用标记好的搜狗语料训练模型  
    
     `python main.py test `   :使用Boson语料进行预测  
    
     `python main.py test taged`   :使用训练好的搜狗语料进行测试  
     
5.建议以下执行顺序：  

    `python main.py test` :直接进行测试
    12月18日，大富科技召开董事会，审议通过了《关于转让全资子公司股权的议案》。大富科技将全资子公司安徽省大富重工技术有限公司（以下简称“大富重工”）100%股权，转让给蚌埠高新投资集团有限公司（以下简称“蚌埠高新投”）

##conda(GPU加速环境):
    wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
    conda install 3.6.5
    conda create -n env4Vtou python=3.6.5
    conda activate env4Vtou
    conda install tensorflow-gpu
    pip install numpy pandas sklearn keras -i https://pypi.doubanio.com/simple
    watch -n1 nvidia-smi
##github:
    git clone https://github.com/changaolin/VTOU.git