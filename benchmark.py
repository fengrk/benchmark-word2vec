# -*- coding:utf-8 -*-
from __future__ import absolute_import

"""
词向量测试 20K

词向量:
- 规模: 19527 x 300D
- 来源: [Chinese-Word-Vectors: sgns.sikuquanshu.word.bz2](https://github.com/Embedding/Chinese-Word-Vectors)

测试结果:
- faiss: load index, 0.82s; search 100 times by word, 1.08s; search 100 times by vec, 1.06s
- gensim: load index, 5.80s; search 100 times by word, 1.64s; search 100 times by vec, 1.62s

"""
import bz2
import logging
import pickle
import time

import gensim
import numpy as np
import os
from pyxtools import global_init_logger, TimeCostHelper
from pyxtools.faiss_tools import faiss


class BasicBenchmark(object):
    """ Basic Class """

    def __init__(self, similar_top_n: int = 20):
        """ init """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similar_top_n = similar_top_n
        self.dimension = None
        self.result_dict = {}
        self.word_vec_model_file = "vec.model"
        self._word_vec_dict = {}

    def prepare(self):
        """ 准备工作 """
        self._global_prepare()

    def _global_prepare(self):
        """  """
        if not os.path.exists(self.word_vec_model_file):
            with open(self.word_vec_model_file, "wb") as fw:
                with bz2.BZ2File('./sgns.sikuquanshu.word.bz2', 'rb') as fr:
                    fw.write(fr.read())

    @staticmethod
    def get_word_list() -> [str]:
        """ 测试词 """
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
        for i in range(1):
            self.search()

        for word in self.get_word_list():
            result_list = self.result_dict[word]
            self.logger.info("{}>>\n{}".format(
                word, "\n".join([result for result in result_list])
            ))
        self.logger.info("Search 1 times by word: cost {} s!".format(time.time() - time_start))

        # search similar words by vec
        self.result_dict.clear()
        time_start = time.time()
        for i in range(1):
            self.vec_search()

        for word in self.get_word_list():
            result_list = self.result_dict[word]
            self.logger.info("{}>>\n{}".format(
                word, "\n".join([result for result in result_list])
            ))
        self.logger.info("Search 1 times by vec: cost {} s!".format(time.time() - time_start))

    def init(self):
        raise NotImplementedError

    def search(self):
        raise NotImplementedError

    def vec_search(self, ):
        raise NotImplementedError

    def save_result_dict(self, word: str, result: str):
        if word not in self.result_dict:
            self.result_dict[word] = [result]
        else:
            result_list = self.result_dict[word]
            if result not in result_list:
                self.result_dict[word].append(result)

    def load_pre_trained_model(self, ):
        """ 返回预训练好的模型 """
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(self.word_vec_model_file, binary=False)
        self.dimension = gensim_model.vector_size
        return gensim_model


class GensimBenchmark(BasicBenchmark):
    """ Gensim """

    def __init__(self):
        super(GensimBenchmark, self).__init__()
        self._model = None

    def init(self):
        self._model = self.load_pre_trained_model()
        for word in self.get_word_list():
            self._word_vec_dict[word] = self._model.get_vector(word)

    def search(self):
        for word in self.get_word_list():
            result = ", ".join([item[0] for item in self._model.similar_by_word(word, topn=self.similar_top_n)])
            self.save_result_dict(word, result)

    def vec_search(self):
        """ 直接使用词向量搜索 """
        for word in self.get_word_list():
            word_vec = self._word_vec_dict[word]
            result = ", ".join(
                [item[0] for item in self._model.similar_by_word(word_vec, topn=self.similar_top_n + 1)[1:]]
            )
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
        gensim_model = self.load_pre_trained_model()
        model_size = len(gensim_model.vocab)
        self.dimension = gensim_model.vector_size
        feature = np.zeros(shape=(model_size, self.dimension), dtype=np.float32)
        word_list = [word for word in gensim_model.vocab]
        for i, word in enumerate(word_list):
            feature[i] = gensim_model.get_vector(word)  # not normed
        self.logger.info("success to load index! Cost {} seconds!".format(time.time() - time_start))

        # train faiss index
        index_factory = "Flat"
        normed_feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        faiss_index = faiss.index_factory(self.dimension, index_factory)
        self.logger.info("training index...")
        time_start = time.time()
        faiss_index.train(normed_feature)  # nb * d
        faiss_index.add(normed_feature)
        self.logger.info("success to train index! Cost {} seconds!".format(time.time() - time_start))

        # save in file
        faiss.write_index(faiss_index, self.faiss_index_file)
        with open(self.faiss_index_detail_pkl, "wb") as f:
            pickle.dump((word_list, feature), f)

    def init(self):
        """ load model """
        self._model = faiss.read_index(self.faiss_index_file)
        with open(self.faiss_index_detail_pkl, "rb") as f:
            self._word_detail_info = pickle.load(f)
            self.dimension = self._word_detail_info[1].shape[-1]

        for word in self.get_word_list():
            self._word_vec_dict[word] = self._word_detail_info[1][self._word_detail_info[0].index(word)]

    def _search_by_vec(self, feature_list, ):
        """ 向量搜索 """
        time_manager = TimeCostHelper()
        normed_feature_list = feature_list / np.linalg.norm(feature_list, axis=1, keepdims=True)
        length = normed_feature_list.shape[0]
        time_manager.dot("_search_by_vec, normed feature , time cost {}s")

        distance_list, indices = self._model.search(normed_feature_list, self.similar_top_n + 1)
        time_manager.dot("_search_by_vec, search , time cost {}s")

        distance_list = distance_list.reshape((length, self.similar_top_n + 1))
        indices = indices.reshape((length, self.similar_top_n + 1))
        time_manager.dot("_search_by_vec, reshape , time cost {}s")

        time_manager.sum("_search_by_vec, total time cost {}s")

        return distance_list, indices

    def search(self):
        """ search similar words """
        # 获取查询词向量
        word_list = self.get_word_list()
        word_feature_list = np.zeros(shape=(len(word_list), self.dimension), dtype=np.float32)
        for i, word in enumerate(word_list):
            word_feature_list[i] = self._word_detail_info[1][self._word_detail_info[0].index(word)]

        # search
        _, indices_arr = self._search_by_vec(word_feature_list)

        # show result
        for i, word in enumerate(word_list):
            result = ", ".join([self._word_detail_info[0][word_index] for word_index in indices_arr[i][1:]])
            self.save_result_dict(word, result)

    def vec_search(self):
        """ 直接使用词向量搜索 """
        # 获取查询词向量
        time_manager = TimeCostHelper()
        word_list = self.get_word_list()
        word_feature_list = np.zeros(shape=(len(word_list), self.dimension), dtype=np.float32)
        for i, word in enumerate(word_list):
            word_feature_list[i] = self._word_vec_dict[word]
        time_manager.dot(info_format="get vec, time cost {}s")

        # search
        _, indices_arr = self._search_by_vec(word_feature_list)
        time_manager.dot(info_format="search by vec, time cost {}s")

        # show result
        for i, word in enumerate(word_list):
            result = ", ".join([self._word_detail_info[0][word_index] for word_index in indices_arr[i][1:]])
            self.save_result_dict(word, result)
        time_manager.dot("get result, time cost {}s")

        time_manager.sum("vec search, time cost {}s")


if __name__ == '__main__':
    # global logger
    global_init_logger()

    # benchmark
    for method_cls in [FaissBenchmark, ]:
        method_cls().run()
