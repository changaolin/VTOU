import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_decode,crf_log_likelihood
from BatchGenerator import BatchGenerator
from corpus import test_input,tagText
import pickle



class Model(object):
    """BILSTM-CRF Model"""
    def __init__(self,dataPath,modelPath='model'):
        super(Model, self).__init__()
        self._load_dataSet(dataPath)
        batch_size = self.config["batch_size"]
        sen_len = self.config['sen_len']
        self.input_data = tf.placeholder(tf.int32, shape=[batch_size, sen_len], name='input')  # 32,60
        self.targets = tf.placeholder(tf.int32, shape=[batch_size, sen_len], name='target')  # 32,60
        self.modelPath = modelPath
        with tf.variable_scope('bilstm_crf') as scope:
            self._build_net()

    def _load_dataSet(self,dataPath):
        with open(dataPath, 'rb') as inp:
            self.word2id = pickle.load(inp)
            self.id2word = pickle.load(inp)
            self.tag2id = pickle.load(inp)
            self.id2tag = pickle.load(inp)
            self.x_train = pickle.load(inp)
            self.y_train = pickle.load(inp)

        self.data_train = BatchGenerator(self.x_train, self.y_train, shuffle=True)
        epochs = 31
        batch_size = 32
        print("train len:", len(self.x_train))
        print("word2id len", len(self.word2id))
        print('Creating the data generator ...')
        print('Finished creating the data generator.')
        config = {}
        config["lr"] = 0.001
        config["emb_dim"] = 100
        config["sen_len"] = len(self.x_train[0])
        config["batch_size"] = batch_size
        config["emb_size"] = len(self.word2id) + 1
        config["tag_size"] = len(self.tag2id)
        config["epochs"] = epochs
        config["dropout_keep"] = 1
        self.config = config
        pass

    def _build_net(self):
        emb_size = self.config['emb_size']
        emb_dim = self.config['emb_dim']
        tag_size = self.config['tag_size']
        dropout_keep = self.config['dropout_keep']
        batch_size = self.config["batch_size"]
        sen_len = self.config['sen_len']
        lr = self.config['lr']
        word_emb = tf.get_variable('word_emb', [emb_size, emb_dim])  # 3448*100
        input_emb = tf.nn.embedding_lookup(word_emb, self.input_data)  # 32*60*100
        input_emb = tf.nn.dropout(input_emb, dropout_keep)

        fw_cell = tf.nn.rnn_cell.LSTMCell(emb_dim, forget_bias=1.0, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.LSTMCell(emb_dim, forget_bias=1.0, state_is_tuple=True)

        # [batch_size, max_time, cell_fw.output_size]  32*60*3448,  32*3448
        (fw_out, bw_out),states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                           bw_cell,
                                                           input_emb,
                                                           dtype=tf.float32,
                                                           time_major=False,
                                                           scope=None)
        # [2, 32] and [2, 2]
        bilstm_out = tf.concat([fw_out, bw_out], axis=2)
        # 32,2*200,10
        W = tf.get_variable(name='W', shape=[batch_size, 2 * emb_dim, tag_size])
        # 32,10,10
        b = tf.get_variable(name='b', shape=[batch_size, sen_len, tag_size], dtype=tf.float32)

        crf_out = tf.tanh(tf.matmul(bilstm_out, W) + b)
        leng = tf.tile(np.array([sen_len]), np.array([batch_size]))
        log_likelihood, self.transition_params = crf_log_likelihood(crf_out, self.targets, leng)
        loss = tf.reduce_mean(-log_likelihood)

        self.viterbi_sequence, score = crf_decode(crf_out, self.transition_params, leng)

        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        self.saver = tf.train.Saver()

    def train(self):
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        batch_num = int(self.data_train.y.shape[0] / batch_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                print("epoch:"+str(epoch))
                print("batch_num:" + str(batch_num))
                for batch in range(batch_num):
                    x_batch, y_batch = self.data_train.next_batch(batch_size)
                    feed_dict = {self.input_data: x_batch, self.targets: y_batch}
                    pre, _ = sess.run([self.viterbi_sequence, self.train_op], feed_dict=feed_dict)
                    acc = 0
                    if batch % 200 == 0:
                        print("epoch:"+str(epoch)+" -- batch:"+str(batch)+" -- batch_len:"+str(len(y_batch)))
                        for i in range(len(y_batch)):
                            for j in range(len(y_batch[0])):
                                if y_batch[i][j] == pre[i][j]:
                                    acc += 1
                        print("epoch:"+str(epoch)+" -- acc_rate:"+str(float(acc) / (len(y_batch) * len(y_batch[0]))))
                if epoch % 3 == 0:
                    path_name = self.modelPath+"/model" + str(epoch) + ".ckpt"
                    print("save", path_name)
                    self.saver.save(sess, path_name)
    def test(self):
        batch_size = self.config['batch_size']
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.modelPath)
            if ckpt is None:
                print("ckpt is None")
                return
            path = ckpt.model_checkpoint_path
            self.saver.restore(sess,path)
            test_input(self,sess,self.word2id,self.id2tag,batch_size)

    def tagText(self,inputP,outP,pre=False):
        batch_size = self.config['batch_size']
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.modelPath)
            if ckpt is None:
                print("ckpt is None")
                return
            path = ckpt.model_checkpoint_path
            self.saver.restore(sess, path)
            tagText(inputP, outP, self, sess, self.word2id, self.id2tag, batch_size,pre)

        pass
    pass

if __name__ == '__main__':
    model = Model()
    # model.train(epochs,data_train)
    # model.test()
    # model.tagText(inputP='TextSub.txt',outP='TextSubTag.txt',pre=False)






















