# -*- coding:utf-8 -*-
"""
词向量测试 6M

词向量:
- 规模: 6115353 x 64D
- 来源:

测试结果:
- faiss: load index, 31.92s; search 100 times by word, 209.59s; search 100 times by vec, 215.94s
- gensim: load index, 208.36s; search 100 times by word, 394.81s; search 100 times by vec, 423.10s

"""
from __future__ import absolute_import

import gensim
from pyxtools import global_init_logger

from benchmark import FaissBenchmark, GensimBenchmark


class Mixin(object):
    def load_pre_trained_model(self, ):
        """ 返回预训练好的模型 """
        return gensim.models.KeyedVectors.load_word2vec_format(self.word_vec_model_file, binary=True)

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


if __name__ == '__main__':
    # global logger
    global_init_logger()

    # benchmark
    for method_cls in [FaissBenchmark1M, GensimBenchmark1M, ]:
        method_cls().run()
