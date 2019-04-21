# benchmark-word2vec
词向量相关词搜索性能对比

## 背景:

实际使用词向量时, 主要使用`gensim`工具提供的`get_vector`与`similar_by_word`接口. 

当模型的词规模在百万级以上时, 调用一次`similar_by_word`, 耗时过长. 如6M大小的模型, 查询单个词的近义词, 耗时约0.35s.

实际项目中, 需要降低检索的耗时. 其中一个解决方案, 便是使用[faiss](https://github.com/facebookresearch/faiss)来实现`similar_by_word`. 

本项目, 简单对比`gensim`与`faiss`在查找近义词方面的性能.

## 实验与结果:

### 20K规模的词向量模型

对应项目中`benchmark.py`, 可通过执行`run.sh`, 获取实验结果.

数据:
- faiss[Flat]: load index, 0.82s; search 100 times by word, 1.08s; search 100 times by vec, 1.06s
- gensim: load index, 5.80s; search 100 times by word, 1.64s; search 100 times by vec, 1.62s

结论: `faiss`在`暴力模式`下运行, 能输出与`gensim`一致的结果, 性能略优于`gensim`.


### 6M规模的词向量模型

对应项目`benchmark_1M.py`.

数据:
- faiss[Flat]: load index, 31.92s; search 100 times by word, 209.59s; search 100 times by vec, 215.94s
- faiss[IMI2x10,Flat; nprobe=8192]: load index, 53.94s; search 100 times by word, 4.36s; search 100 times by vec, 4.22s; train+store, 197.05s
- gensim: load index, 208.36s; search 100 times by word, 394.81s; search 100 times by vec, 423.10s

结论:
- `faiss`在`暴力模式`下运行, 能输出与`gensim`一致的结果. 耗时约为`gensim`的0.5倍
- `faiss`在`IMI2x10,Flat`模式下, 训练耗时约200s. 通过提高查询时的`nprobe`, 能提高检索召回率. 当`nprobe=256`, 检索结果可以接受; 当`nprobe=8192`, 基本与`gensim`一致, 但耗时仅为`gensim`的0.01倍. 

## 感谢:

- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)分享的[预训练的词向量模型](https://pan.baidu.com/s/1vPSeUsSiWYXEWAuokLR0qQ)(见项目内的`sgns.sikuquanshu.word.bz2`文件)
- [自然语言处理中句子相似度计算的几种方法](https://cuiqingcai.com/6101.html)提供的[预训练模型](https://pan.baidu.com/s/1TZ8GII0CEX32ydjsfMc0zw)
- [faiss项目](https://github.com/facebookresearch/faiss)
- [onfido/faiss_prebuilt](https://github.com/onfido/faiss_prebuilt)
- [gensim](https://radimrehurek.com/gensim/)
