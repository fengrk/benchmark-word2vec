# -*- coding:utf-8 -*-
from __future__ import absolute_import

import bz2
import logging
import pickle
import time
from pyxtools import global_init_logger,

import gensim
import numpy as np
import os
from pyxtools.faiss_tools import faiss

word_vec_model_file = "vec.model"
if not os.path.exists(word_vec_model_file):
    with open(word_vec_model_file, "wb") as fw:
        with bz2.BZ2File('./sgns.sikuquanshu.word.bz2', 'rb') as fr:
            fw.write(fr.read())


class BasicBenchmark(object):
    """ Basic Class """

    def __init__(self, similar_top_n: int = 20):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similar_top_n = similar_top_n
        self.dimension = 300
        self.result_dict = {}

    def prepare(self):
        """ 准备工作 """
        pass

    @staticmethod
    def get_word_list() -> [str]:
        return ["计", "算", "机", "词", "向", "量", "囧"]

    def run(self):
        # prepare
        self.prepare()

        # init
        time_start = time.time()
        self.init()
        self.logger.info("Init: cost {} s!".format(time.time() - time_start))

        # search similar words
        time_start = time.time()
        for i in range(100):
            self.search()

        for word in self.get_word_list():
            result_list = self.result_dict[word]
            self.logger.info("{}>>\n{}".format(
                word, "\n".join([result for result in result_list])
            ))
        self.logger.info("Search 100 times: cost {} s!".format(time.time() - time_start))

    def init(self):
        raise NotImplementedError

    def search(self):
        raise NotImplementedError

    def save_result_dict(self, word: str, result: str):
        if word not in self.result_dict:
            self.result_dict[word] = [result]
        else:
            result_list = self.result_dict[word]
            if result not in result_list:
                self.result_dict[word].append(result)


class GensimBenchmark(BasicBenchmark):
    """ Gensim """

    def __init__(self):
        super(GensimBenchmark, self).__init__()
        self._model = None

    def init(self):
        self._model = gensim.models.KeyedVectors.load_word2vec_format(word_vec_model_file, binary=False)

    def search(self):
        for word in self.get_word_list():
            result = ", ".join([item[0] for item in self._model.similar_by_word(word, topn=self.similar_top_n)])
            self.save_result_dict(word, result)


class FaissBenchmark(BasicBenchmark):
    """ Faiss """

    def __init__(self):
        super(FaissBenchmark, self).__init__()
        self._model = None
        self._word_detail_info = None
        self.faiss_index_file = "./faiss.index"
        self.faiss_index_detail_pkl = "./faiss.pkl"

    def prepare(self):
        """ 将Gensim 版本的模型转化为Faiss模型 """
        super(FaissBenchmark, self).prepare()

        # turn model from gensim to faiss index
        if os.path.exists(self.faiss_index_file) and os.path.exists(self.faiss_index_detail_pkl):
            return

        # load model to dict
        self.logger.info("loading model...")
        time_start = time.time()
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(word_vec_model_file, binary=False)
        model_size = len(gensim_model.vocab)
        assert self.dimension == gensim_model.vector_size
        feature = np.zeros(shape=(model_size, self.dimension), dtype=np.float32)
        word_list = [word for word in gensim_model.vocab]
        for i, word in enumerate(word_list):
            feature[i] = gensim_model.get_vector(word)
        self.logger.info("success to load index! Cost {} seconds!".format(time.time() - time_start))

        # train faiss index
        index_factory = "Flat"
        faiss_index = faiss.index_factory(self.dimension, index_factory)
        self.logger.info("training index...")
        time_start = time.time()
        faiss_index.train(feature)  # nb * d
        self.logger.info("success to train index! Cost {} seconds!".format(time.time() - time_start))
        faiss_index.add(feature)

        # save in file
        faiss.write_index(faiss_index, self.faiss_index_file)
        with open(self.faiss_index_detail_pkl, "wb") as f:
            pickle.dump((word_list, feature), f)

    def init(self):
        """ load model """
        self._model = faiss.read_index(self.faiss_index_file)
        with open(self.faiss_index_detail_pkl, "rb") as f:
            self._word_detail_info = pickle.load(f)

    def search(self):
        """ search similar words """

        def search_by_vec(feature_list, ):
            """ 向量搜索 """
            length = feature_list.shape[0]

            time_start = time.time()
            distance_list, indices = self._model.search(feature_list, self.similar_top_n)
            self.logger.info("cost {}s to search {} feature".format(time.time() - time_start, length))

            distance_list = distance_list.reshape((length, self.similar_top_n))
            indices = indices.reshape((length, self.similar_top_n))

            return distance_list, indices

        # 获取查询词向量
        word_list = self.get_word_list()
        word_feature_list = np.zeros(shape=(len(word_list), self.dimension), dtype=np.float32)
        for i, word in enumerate(word_list):
            word_feature_list[i] = self._word_detail_info[1][self._word_detail_info[0].index(word)]

        # search
        _, indices_arr = search_by_vec(word_feature_list)

        # show result
        for i, word in enumerate(word_list):
            result = ", ".join([self._word_detail_info[0][word_index] for word_index in indices_arr[i]])
            self.save_result_dict(word, result)


if __name__ == '__main__':
    # global logger
    global_init_logger()

    # benchmark
    for method_cls in [FaissBenchmark, GensimBenchmark, ]:
        method_cls().run()
