#!/usr/bin/env python
# coding:utf-8

import sys
import jieba
import json
import numpy as np
import datetime

from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, BatchNormalization, TimeDistributed, Lambda, Flatten, RepeatVector, Permute, Activation, merge, Dropout
from keras.models import Model, load_model
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers.merge import concatenate
from keras import backend as K

sDay = datetime.date.today().strftime('%Y%m%d')

class StaticHistory(Callback):

    def __init__(self, test_data, test_label):
        self.logfile = "/data/ceph/query/multi/train/scripts/logs/multilogs"
        self.test_data = test_data
        self.test_label = test_label
        self.savefile = "/data/ceph/query/multi/train/model/model_multi_" + sDay + "_best.hdf5"
 
    def on_train_begin(self, logs={}):
        self.acc_level1_max = 0 
        self.acc_level2_max = 0
    def on_epoch_end(self, epoch, logs={}):
        [output_class1, output_class2] = self.model.predict(self.test_data, batch_size=256)
        preds_level1 = np.argmax(output_class1, axis=1)
        preds_level2 = np.argmax(output_class2, axis=1)
        self.acc_level1 = logs.get('val_level1_acc')
        self.acc_level2 = logs.get('val_level2_acc')
        with open(self.logfile, "a") as wf:
            wf.write("epoch " + str(epoch) + "\t")
            wf.write(classification_report(self.test_label[0], preds_level1))
            wf.write(classification_report(self.test_label[1], preds_level2))
            wf.write("level1_loss " + str(logs.get("level1_loss")) + "\t")
            wf.write("level2_loss " + str(logs.get("level2_loss")) + "\t")
            wf.write("level1_acc " + str(logs.get("level1_acc")) + "\t")
            wf.write("level2_acc " + str(logs.get("level2_acc")) + "\t")
            wf.write("val_level1_loss " + str(logs.get("val_level1_loss")) + "\t")
            wf.write("val_level2_loss " + str(logs.get("val_level2_loss")) + "\t")
            wf.write("val_level1_acc " + str(logs.get("val_level1_acc")) + "\t")
            wf.write("val_level2_acc " + str(logs.get("val_level2_acc")) + "\n")
            if self.acc_level2_max < self.acc_level2:
                wf.write("model saved after epoch " + str(epoch) + "\t")
                self.model.save(self.savefile)
                self.acc_level2_max = self.acc_level2

class queryMultiClassifier(object):

    def __init__(self):

        self.datapath = "/data/ceph/query/multi/train/data/positive"
        self.dictfile = "/data/ceph/query/multi/train/data/vocab_word" + sDay + ".json"
        self.savefile = "/data/ceph/query/multi/train/model/model_multi_word_" + sDay + ".hdf5"
        self.features_dim = 100
        self.max_length = 10
        self.hidden_dims = 100
        self.NUM_CLASSES1 = 17
        self.NUM_CLASSES2 = 80

    def preprocessing(self):

        self.cate2tokens = {}
        self.allTokenList = []
        self.labels = []
        cate2Querys = {}

        with open(self.datapath, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                contents = line.split("\t")
                query = contents[0]
                cates = contents[1]
                if (cates not in cate2Querys.keys()):
                    cate2Querys[cates] = [query]
                else:
                    (cate2Querys[cates]).append(query)

        for (key, value) in cate2Querys.items():
            tokens = set()
            label_1 = key[:2]
            label_2 = key[3:]
            for word in value:
                tokenList = []
                tokenIters = jieba.cut(word)
                for token in tokenIters:
                    tokens.add(token)
                    tokenList.append(token)
                query2labels = []
                print word
                query2labels.append(int(label_1))
                query2labels.append(int(label_2))
                self.allTokenList.append(tokenList)
                self.labels.append(query2labels)
            self.cate2tokens[key] = tokens

    def word2vec(self):
        texts = []
        for (key, value) in self.cate2tokens.items():
            texts.append(value)
        model = Word2Vec(texts, self.features_dim, min_count=1)
        self.w2v = model.wv

    def saveDict(self):
        with open(self.dictfile, "w") as f:
            json.dump(self.dictionary, f)

    def getDict(self):
        self.dictionary = defaultdict()
        for index, word in enumerate(self.w2v.index2word):
            self.dictionary[word] = index

    def replaceWordbyID(self):
        self.new_docs = []
        for text in self.allTokenList:
            new_text = []
            for word in text:
                new_text.append(self.dictionary[word])
            self.new_docs.append(np.array(new_text))
        del self.allTokenList

    def pad_seq(self):
        self.sequences = pad_sequences(sequences=self.new_docs, maxlen=self.max_length)
        del self.new_docs

    def split_data(self):
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(self.sequences, self.labels, test_size=0.2, random_state=1)
        train_label_temp = np.array(self.train_label).transpose()
        self.train_label = list(train_label_temp)
        test_label_temp = np.array(self.test_label).transpose()
        self.test_label = list(test_label_temp)
        del self.sequences, self.labels

    def step_decay(self, epoch):
        if epoch % 4 == 3:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.5) 
            print ("lr changed to {}".format(lr * 0.5))
        return K.get_value(self.model.optimizer.lr)       

    def build_network(self):

        print("Start to build the DL model")

        embedder = Embedding(input_dim=len(self.w2v.index2word),
                                                output_dim=self.features_dim,
                                                weights=[self.w2v.syn0],
                                                trainable=True)
        sequence_input = Input(shape=(self.max_length,), dtype='int32')
        embedded_sequences = embedder(sequence_input)

        x = Bidirectional(LSTM(self.hidden_dims, dropout_W=0.2, dropout_U=0.2, return_sequences=True), merge_mode='concat')(embedded_sequences)

        attention = TimeDistributed(Dense(1, activation='tanh'))(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_dims * 2)(attention)  # biDirection
        attention = Permute((2, 1))(attention)         

        x = merge([x, attention], mode='mul')
        x = Lambda(lambda xx: K.sum(xx, axis=1))(x)
        x = Dense(self.hidden_dims, activation="sigmoid")(x)
        x = BatchNormalization()(x)
        x = Dense(self.hidden_dims, activation='sigmoid')(x)
        x = BatchNormalization()(x)

        output_class1 = Dense(self.NUM_CLASSES1, activation="softmax", name='level1')(x)
        output_dropout = Dropout(0.5)(output_class1)
        output_class1_concatenate = concatenate([x, output_dropout])
        output_class2 = Dense(self.NUM_CLASSES2, activation="softmax", name='level2')(output_class1_concatenate)

        self.model = Model(sequence_input, [output_class1, output_class2])

        self.model.compile (optimizer="Adadelta",
                                        loss='sparse_categorical_crossentropy',
                                        loss_weights={"level1": 5., "level2": 5.},
                                        metrics=["accuracy"])
        print(self.model.summary())
        print("Get the model build work Done!")

    def train(self, num_epochs):
        self.preprocessing()
        self.word2vec()
        self.getDict()
        self.saveDict()
        self.replaceWordbyID()
        self.pad_seq()
        self.split_data()
        self.build_network()
#        self.model = load_model(self.savefile)
        self.static_history = StaticHistory(self.test_data, self.test_label)
        K.set_value(self.model.optimizer.lr, 0.1)
        lrate = LearningRateScheduler(self.step_decay)
        callback_list = [lrate, self.static_history]
        self.model.fit(self.train_data, self.train_label,
                               validation_data=(self.test_data, self.test_label),
                               batch_size=256,
                               epochs=num_epochs,
                               callbacks=callback_list,
                               verbose=1)
        self.model.save(self.savefile)
        del self.model

if __name__ == "__main__":
    num_epochs = int(sys.argv[1])
    clf = queryMultiClassifier()
    clf.train(num_epochs)
