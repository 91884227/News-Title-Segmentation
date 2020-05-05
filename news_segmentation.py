#!/usr/bin/env python
# coding: utf-8

# # import package

# In[1]:


import pandas as pd
import json
import numpy as np
import itertools
import collections
import tensorflow as tf
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import os
from datetime import date
from IPython.display import clear_output
clear_output()
from tqdm import tqdm
import sys


# In[2]:


if __name__ == '__main__':
#     FILENAME = "katino_data_adjust.npy"
#     WORD_TO_WEIGHT = "斷詞詞庫.txt"
#     LIMIT = 1
#     CUDA_VISIBLE_DEVICES = "0"
#     GPU_MEMORY_FRACTION = 0.7
    FILENAME =  sys.argv[1]
    WORD_TO_WEIGHT = sys.argv[2]
    LIMIT = int(sys.argv[3])
    CUDA_VISIBLE_DEVICES = str(sys.argv[4])
    GPU_MEMORY_FRACTION = float(sys.argv[4])

    print("set GPU stat...")
    cfg = tf.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction =  GPU_MEMORY_FRACTION ###設定gpu使用量
    session = tf.Session(config = cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES ###設定gpu編號

    print("prepare ws pos ner")
    ws = WS("./data", disable_cuda=False)
    pos = POS("./data", disable_cuda=False)
    ner = NER("./data", disable_cuda=False)

    print("read data in...")
    data = np.load(FILENAME)
    if( LIMIT ):
        data = data[:1000]

    print("read WORD_TO_WEIGHT in...")
    word_to_weight = {}
    with open(WORD_TO_WEIGHT, encoding='utf-8') as f:
        for line in f:
            word = line.split('\n')[0]
            if(word not in word_to_weight):
                word_to_weight[word]=1
            else:
                word_to_weight[word]+=1
    dictionary = construct_dictionary(word_to_weight)

    print("start segementation...")
    word_sentence_list = ws(
        data,
        sentence_segmentation = True, # To consider delimiters
        # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
        # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
        # coerce_dictionary = dictionary2, # words in this dictionary are forced
    )


    print("start POS...")
    pos_sentence_list = pos(word_sentence_list)

    print("start to save the result...")
    savename = "%s_ws.json" % FILENAME[:-4]
    with open(savename, "w") as fp:
        json.dump(word_sentence_list, fp)
        print("save as %s" % (savename))

    savename = "%s_POS.json" % FILENAME[:-4]
    with open(savename, "w") as fp:
        json.dump(pos_sentence_list, fp)
        print("save as %s" % (savename))


# In[ ]:




