#conda:
    wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
    conda install 3.6.5
    conda create -n env4Vtou python=3.6.5
    conda activate env4Vtou
    conda install tensorflow-gpu
    pip install numpy pandas sklearn -i https://pypi.doubanio.com/simple
    watch -n1 nvidia-smi
#github:
    git clone https://github.com/changaolin/VTOU.git
# VTOU
实验内容：  

1.处理标注语料：[BosonNLP_NER_6C.txt](https://bosonnlp.com/resources/BosonNLP_NER_6C.zip)  

2.预训练实体标注模型  

3.
搜狗实验室语料-BosonNLP_NER_6C.txt-BILSTM-CRF-NER

