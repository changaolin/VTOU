import sys,os
from corpus import preText,savePkl
from model import Model
from ext import sendtoPhone
data_base = 'corpus'
data_pre_name = 'BosonNLP_NER_6C.txt'
data_pre_pkl = 'BosonNLP.pkl'
pre_model_path = './preModel'
data_tag_ori_file = 'shangye.txt'
data_tag_out_file = "Tag.pkl"
taged_model_path = './TagedModel'
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('usage:python3 main.py [prepare|train|tag|test]')
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'prepare':
            # 处理预料，预训练得到初始模型用于进行序列标注任务
            preText(data_pre_name,'result.txt',data_base)
            savePkl('result.txt',data_pre_pkl,data_base)
            model = Model(dataPath=os.path.join(data_base,data_pre_pkl),modelPath=pre_model_path)
            try:
                model.train()
                sendtoPhone('prepare down！')
            except Exception as e:
                print("error")
                sendtoPhone('prepare ERROR！')

        elif sys.argv[1] == 'train':
            dataPath = data_pre_pkl
            modelPath = pre_model_path
            if len(sys.argv) > 3:
                dataPath = sys.argv[2]
                modelPath = sys.argv[3]
            elif sys.argv[2] == 'taged':
                dataPath = data_tag_out_file
                modelPath = taged_model_path
                print("dataPath:"+dataPath)
                print("modelPath:" + modelPath)
            model = Model(dataPath=os.path.join(data_base, data_pre_pkl), modelPath=pre_model_path)
            try:
                model.train()
                sendtoPhone('train down！')
            except Exception as e:
                print("error")
                sendtoPhone('train ERROR！')
        elif sys.argv[1] == 'test':
            if len(sys.argv) == 2:
                model = Model(dataPath=os.path.join(data_base, data_pre_pkl), modelPath=pre_model_path)
                model.test()
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
                print("...")
                # model.test(inputP=inputP,outP=outP,pre=False)
                # sendtoPhone('test down！')
            except Exception as e:
                print(e)
                # sendtoPhone("test Error")

        elif sys.argv[1] == 'tag':
            if len(sys.argv) > 3:
                inp = sys.argv[2]
                oup = sys.argv[3]
            else:
                inp = data_tag_ori_file
                oup = data_tag_out_file
            model = Model(dataPath=os.path.join(data_base,data_pre_pkl),modelPath=pre_model_path)
            inputP = os.path.join(data_base,data_tag_ori_file)
            outP = os.path.join(data_base,data_tag_out_file)
            try:
                model.tagText(inputP=inputP, outP=outP, pre=False)
                sendtoPhone('tag down！')
            except Exception as e:
                print(e)
                sendtoPhone("tag Error")

    pass