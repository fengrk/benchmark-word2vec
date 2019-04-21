# -*- coding:utf-8 -*-
"""
词向量测试 6M

词向量:
- 规模: 6115353 x 64D
- 来源: [自然语言处理中句子相似度计算的几种方法](https://cuiqingcai.com/6101.html)提供的[news_12g_baidubaike_20g_novel_90g_embedding_64.bin](https://pan.baidu.com/s/1TZ8GII0CEX32ydjsfMc0zw)

使用(下面代码可能有错误, 请自行修改):

```
# 编辑Dockerfile
echo 'FROM frkhit/benchmark-word2vec:latest
COPY news_12g_baidubaike_20g_novel_90g_embedding_64.bin ./word_vec_1m.bin
COPY benchmark_1M.py benchmark.py ./
ENTRYPOINT ["python"]
CMD ["-u", "benchmark_1M.py"]
' > Dockerfile

# docker build
docker build -t benchmark:latest .

# docker run
docker logs -f $(docker run -d benchmark:latest)
```

测试结果:
- faiss[Flat]: load index, 31.92s; search 100 times by word, 209.59s; search 100 times by vec, 215.94s
- faiss[IMI2x10,Flat; nprobe=8192]: load index, 53.94s; search 100 times by word, 4.36s; search 100 times by vec, 4.22s
- gensim: load index, 208.36s; search 100 times by word, 394.81s; search 100 times by vec, 423.10s

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


class FaissBenchmark1M(Mixin, FaissBenchmark):
    """ Faiss 1M words"""

    def __init__(self):
        super(FaissBenchmark1M, self).__init__()
        self.word_vec_model_file = "word_vec_1m.bin"
        self.faiss_index_file = "./faiss_1m.index"
        self.faiss_index_detail_pkl = "./faiss_1m.pkl"

        # faiss setting
        self._faiss_factory = "IMI2x10,Flat"
        self.n_probe = 8192

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
        normed_feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        faiss_index = faiss.index_factory(self.dimension, self._faiss_factory)
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
    for method_cls in [FaissBenchmark1M, GensimBenchmark1M, ]:
        method_cls().run()
