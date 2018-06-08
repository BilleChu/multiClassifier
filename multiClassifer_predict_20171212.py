#! /usr/bin/env python
# coding: utf-8

import sys
import jieba
import json
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
#from keras.utils import plot_model

sDay = ""

class queryMultiClassifier(object):

    def __init__(self):

        self.filename = "/data/ceph/query/binary/data/predict/result/preds_highBelief_pos_" + sDay
        self.dictfile = "/data/ceph/query/multi/train/data/vocab_word.json"
        self.mapfile = "/data/ceph/query/multi/train/data/mapCates"
        self.modelfile = "/data/ceph/query/multi/train/model/model_multi_word.hdf5"
        self.savefile = "/data/ceph/query/multi/predict/data/pred_" + sDay
        self.max_length = 10
        self.NUM_CLASSES1 = 17
        self.NUM_CLASSES2 = 80
        self.thresh_class1 = 0.4
        self.thresh_calss2 = 0.4
        self.batch_size = 128

    def __load_map__(self):
        self.map_category_1 = {}
	self.map_category_2 = {}
        with open(self.mapfile) as f:
            lines = f.readlines()
            for line in lines:
                contents = line.split("  ")
                if int(contents[0]) not in self.map_category_1:
                    self.map_category_1[int(contents[0])] = contents[2].strip()
                if int(contents[1]) not in self.map_category_2:
                    self.map_category_2[int(contents[1])] = contents[3].strip()
                
    def preprocessing(self):
#        print("preprocessing...")
        self.corpus = []
        self.queryList = []
        with open(self.filename, "r") as f:
            i = 0
            while (True):
                line = f.readline()
                if not line:
                    break
                query = line.strip()
                tokenList = []
                tokenIters = jieba.cut(query)
                for token in tokenIters:
                    token = token.decode('utf-8', 'ignore')
                    if (token in self.dictionary):
                        tokenList.append(self.dictionary[token])
                    else:
                        continue
                if (len(tokenList) != 0):
                    self.corpus.append(tokenList)
                    self.queryList.append(query)
                i += 1
                if i%self.batch_size == 0:
                    print i
                    self.pad_data()
                    self.predict()
                    self.corpus = []
                    self.queryList = [] 

    def loadDict(self):
        print("loadDict...")
        with open(self.dictfile, "r") as f:
            self.dictionary = json.load(f)

    def pad_data(self):
#        print("padding")
        self.sequences = pad_sequences(sequences=self.corpus, maxlen=self.max_length)

    def __load_model__(self):
        print("load_model...")
        self.model = load_model(self.modelfile)

    def predict(self):
#        print ("ready to predict...")
        [output1, output2] = self.model.predict(self.sequences, batch_size=self.batch_size)
        label_1 = np.argmax(output1, axis=1)
        weight_1 = np.max(output1, axis=1)
        label_2 = np.argmax(output2, axis=1)
        weight_2 = np.max(output2, axis=1)
        print ("predict finished...")
        with open(self.savefile, "a") as f:
            for index ,text in enumerate(self.queryList):
                if weight_1[index] > self.thresh_class1 and weight_2[index] > self.thresh_class2:
                    f.write(text + "|" + self.map_category_1[label_1[index]] + "|" +self.map_category_2[label_2[index]] + "\n")

    def call(self):
        self.loadDict()
        self.__load_model__()
        self.__load_map__()
        self.preprocessing()
#        plot_model(self.model, to_file("model.png"))
if __name__ == "__main__":
    sDay = sys.argv[1]
    clf = queryMultiClassifier()
    clf.max_length = int(sys.argv[2])
    clf.batch_size = int(sys.argv[3])
    clf.thresh_class1 = float(sys.argv[4])
    clf.thresh_class2 = float(sys.argv[5])
    clf.call()
