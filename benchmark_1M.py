# -*- coding:utf-8 -*-
"""
词向量测试 6M

词向量:
- 规模: 6115353 x 64D
- 来源:

测试结果:
- faiss: load index, 31.92s; search 100 times by word, 209.59s; search 100 times by vec, 215.94s
- gensim: load index, 208.36s; search 100 times by word, 394.81s; search 100 times by vec, 423.10s

优化测试:
- Flat: index文件大小 1.5GB;  训练+存储 5.58s; 加载index 42.99s; 搜索词列表1次耗时 2.4s; 结果精确, 与gensim一致
- IMI2x10,Flat: 聚类分桶->存储原始向量. index文件大小 1.6GB; 训练+存储 197.05s; 加载index 43.22s; 搜索词列表1次耗时 0.004s; 准确度低
    例如: 未来 >> 重要, 有利, 重视, 看重, 可靠, 心就行, 誉比, 丰厚, 有价值, 而且, 优厚, 低廉, 高圳气, 高圳气, 高圳气, 高圳气, 高圳气, 高圳气, 高圳气, 高圳气

    修改nprobe:
        - 1, 0.3s
        - 8, 0.04s
        - 32, 0.005s
        - 256, 0.008s, 准确率可以接受
        - 1024, 0.016s
        - 2048, 0.028s
        - 8196, 0.045s, 准确率非常高, 基本等同于gensim
"""
from __future__ import absolute_import

import pickle
import time

import gensim
import numpy as np
import os
from pyxtools import global_init_logger
from pyxtools.faiss_tools import faiss

from benchmark import FaissBenchmark, GensimBenchmark

word_vec_pkl = "./big.pkl"


class Mixin(object):
    def load_pre_trained_model(self, ):
        """ 返回预训练好的模型 """
        if not os.path.exists(word_vec_pkl):
            model = gensim.models.KeyedVectors.load_word2vec_format(self.word_vec_model_file, binary=True)
            with open(word_vec_pkl, "wb") as fw:
                pickle.dump(model, fw)

        with open(word_vec_pkl, "rb") as fr:
            return pickle.load(fr)

    def _global_prepare(self):
        """  """
        pass

    @staticmethod
    def get_word_list() -> [str]:
        """ 测试词 """
        return ["计算机", "中国", "人工智能", "自然语言", "语言", "科学", "哲学", "未来", "人类", "地球"]


class GensimBenchmark1M(Mixin, GensimBenchmark):
    """ Gensim 1M words"""

    def __init__(self):
        super(GensimBenchmark1M, self).__init__()
        self.word_vec_model_file = "word_vec_1m.bin"
        self.dimension = 64


class FaissBenchmark1M(Mixin, FaissBenchmark):
    """ Faiss 1M words"""

    def __init__(self):
        super(FaissBenchmark1M, self).__init__()
        self.word_vec_model_file = "word_vec_1m.bin"
        self.faiss_index_file = "./faiss_1m.index"
        self.faiss_index_detail_pkl = "./faiss_1m.pkl"
        self.dimension = 64
        self.n_probe = 1

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
        index_factory = "IMI2x10,Flat"
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

    def search(self):
        """ search similar words """
        self._model.nprobe = self.n_probe
        super(FaissBenchmark1M, self).search()

    def vec_search(self):
        """ 直接使用词向量搜索 """
        self._model.nprobe = self.n_probe
        super(FaissBenchmark1M, self).vec_search()


if __name__ == '__main__':
    # global logger
    global_init_logger()

    # benchmark
    for method_cls in [GensimBenchmark1M]:
        method_cls().run()

    faiss_obj = FaissBenchmark1M()
    for n_probe in [1, 8, 32, 256, 1024, 2048, 8196]:
        faiss_obj.logger.info("Next>>\n\n\nn_probe is {}".format(n_probe))
        faiss_obj.n_probe = n_probe
        faiss_obj.run()
