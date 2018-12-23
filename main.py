import sys,os
from corpus import preText,savePkl
from model import Model
from ext import sendtoPhone
from utils import taLogging
if not os.path.exists('log'):
    os.mkdir('log')
logger = taLogging.getFileLogger(name='main',level=1,file='log/main.log')
data_base = 'corpus'
data_pre_name = 'BosonNLP_NER_6C.txt'
data_pre_pkl = 'BosonNLP.pkl'
pre_model_path = './preMoodel'
data_tag_ori_file = 'shangyeSub.txt'
data_tag_out_file = "shangyeTag.tag"
data_tag_out_pkl = "shangyeTag.pkl"
taged_model_path = './TagedModel'

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('usage:python3 main.py [prepare|train|tag|test]')
    if len(sys.argv) >= 2:
        # 用boson语料训练模型
        if sys.argv[1] == 'prepare':
            logger.debug("start prepare")
            # 处理预料，预训练得到初始模型用于进行序列标注任务
            preText(data_pre_name,'result.txt',data_base)
            savePkl('result.txt',data_pre_pkl,data_base)
            model = Model(dataPath=os.path.join(data_base,data_pre_pkl),modelPath=pre_model_path)
            try:
                model.train()
                sendtoPhone('prepare-down')
                logger.debug("prepare-down！")
            except Exception as e:
                print("error")
                sendtoPhone('prepare-ERROR')
                logger.debug("prepare-ERROR！")

        elif sys.argv[1] == 'train':
            # 训练模型，taged 表示使用标注的搜狗语料训练
            dataPath = data_pre_pkl
            modelPath = pre_model_path
            if len(sys.argv) > 3:
                dataPath = sys.argv[2]
                modelPath = sys.argv[3]
            elif sys.argv[2] == 'taged':
                dataPath = data_tag_out_pkl
                modelPath = taged_model_path
                logger.debug("dataPath:"+dataPath)
                logger.debug("modelPath:" + modelPath)
                print("dataPath:"+dataPath)
                print("modelPath:" + modelPath)
            model = Model(dataPath=os.path.join(data_base, dataPath), modelPath=modelPath)
            try:
                model.train()
                sendtoPhone('train-down')
                logger.debug("train down")
            except Exception as e:
                print("error")
                sendtoPhone('train-ERROR')
                logger.debug("train ERROR！")

        elif sys.argv[1] == 'test':
            # 测试模型，taged 表示使用标注的搜狗语料测试
            dataPath = data_pre_pkl
            modelPath = pre_model_path
            if len(sys.argv) > 3:
                dataPath = sys.argv[2]
                modelPath = sys.argv[3]
            elif sys.argv[2] == 'taged':
                dataPath = data_tag_out_pkl
                modelPath = taged_model_path
                logger.debug("dataPath:" + dataPath)
                logger.debug("modelPath:" + modelPath)
                print("dataPath:" + dataPath)
                print("modelPath:" + modelPath)
            model = Model(dataPath=os.path.join(data_base, dataPath), modelPath=modelPath)
            model.test()

        elif sys.argv[1] == 'tag':
            # 使用boson语料训练的模型对搜狗语料进行标注，加大数据量，可能会引入较大的误差，待优化
            if len(sys.argv) > 3:
                inp = sys.argv[2]
                oup = sys.argv[3]
            else:
                inp = data_tag_ori_file
                oup = data_tag_out_file
            model = Model(dataPath=os.path.join(data_base,data_pre_pkl),modelPath=pre_model_path)
            inputP = os.path.join(data_base,inp)
            outP = os.path.join(data_base,oup)
            try:
                inf = {
                    "input":inputP,
                    "output":outP
                }
                logger.debug(inf)
                model.tagText(inputP=inputP, outP=outP, pre=False)
                savePkl(oup, data_tag_out_pkl, data_base)
                sendtoPhone('tag-down')
                logger.debug("tag-down")
            except Exception as e:
                print(e)
                sendtoPhone("tag-ERROR")
                logger.debug("tag-ERROR！")
    pass