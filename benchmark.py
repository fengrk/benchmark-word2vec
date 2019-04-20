# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import time
from bz2 import BZ2File

import gensim
import os

word_vec_model_file = "vec.model"
if not os.path.exists(word_vec_model_file):
    with open(word_vec_model_file, "wb") as fw:
        with BZ2File('./sgns.sikuquanshu.word.bz2', 'rb') as fr:
            fw.write(fr.read())


class BasicBenchmark(object):
    def __init__(self, similar_top_n: int = 20):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similar_top_n = similar_top_n

    @staticmethod
    def get_word_list() -> [str]:
        return ["计", "算", "机", "词", "向", "量", "囧"]

    def run(self):
        # init
        time_start = time.time()
        self.init()
        self.logger.error("Init: cost {} s!".format(time.time() - time_start))

        # search similar words
        time_start = time.time()
        for i in range(100):
            self.search()
        self.logger.error("Search 100 times: cost {} s!".format(time.time() - time_start))

    def init(self):
        raise NotImplementedError

    def search(self):
        raise NotImplementedError


class GensimBenchmark(BasicBenchmark):
    def __init__(self):
        super(GensimBenchmark, self).__init__()
        self._model = None

    def init(self):
        self._model = gensim.models.KeyedVectors.load_word2vec_format(word_vec_model_file, binary=False)

    def search(self):
        for word in self.get_word_list():
            self.logger.error("{}: {}".format(
                word, ", ".join([item[0] for item in self._model.similar_by_word(word, topn=self.similar_top_n)])
            ))


if __name__ == '__main__':
    for method_cls in [GensimBenchmark, ]:
        method_cls().run()
