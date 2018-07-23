#!/usr/bin/python
# -*- coding:utf-8 -*-
# Copyright 2017, Baidu Inc.
# Author: wangqi

# description: 预处理模块 主要包括构建词典

import sys
import os
import numpy
import joblib
import jieba
from collections import OrderedDict
import pandas as pd
def make_csv(filename):
    name = ['label','query','title']
    df = pd.read_table(filename,names=name)
    df['query_jb'] = df['query'].apply(word_seg)
    df['titile_jb'] = df['title'].apply(word_seg)
    # items[1] = remove_stop_words(items[1])
    # items[2] = remove_stop_words(items[2])


#将训练数据分开成三种文件
def split_query_title_label(filename):
    query_writer = open(filename+'.jieba.query','w')
    title_writer = open(filename + '.jieba.title', 'w')
    label_writer = open(filename + '.jieba.label', 'w')
    word2vec_writer = open(filename + '.w2v', 'w')
    for line in open(filename, 'r'):
        items = line.strip().split('\t')
        # items[1] = remove_stop_words(items[1])
        # items[2] = remove_stop_words(items[2])
        str1 = word_seg(items[1])
        str2 = word_seg(items[2])
        query_writer.write(str1+'\n')
        title_writer.write(str2+'\n')
        label_writer.write(items[0]+'\n')
        word2vec_writer.write(str1+" "+str2+'\n')
    query_writer.close()
    title_writer.close()
    label_writer.close()

def remove_stop_words(item):
    wordlst = item.split(' ')
    stopwords = load_stopword('../const_file/stop_words_small.txt')
    new_seg = [x for x in wordlst if x not in stopwords]
    return ' '.join(new_seg)

def load_stopword(filename):
    stopwords = {x.strip() for x in open(filename, 'r').readlines()}
    return stopwords

def symbol_seg(line):
    line = line.replace(" ",'')
    return ' '.join(list(line))

def word_seg(line):
    #jieba.load_userdict('dict')
    #stopwords = load_stopword('./const_file/stop_word.txt')
    seg_list = jieba.cut(line.replace(" ",''))
    #new_seg = [x for x in seg_list if x not in stopwords]
    new_seg = [x.lower() for x in seg_list if x.strip() != '']
    return ' '.join(new_seg)


#构建词典
def build_dictionary(file_lst, dst_path,cut_num):
    word_freqs = OrderedDict()
    for filename in file_lst:
        for line in open(filename,'r'):
            print(line)
            words_in = line.strip().split(' ')
            for w in words_in:
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w] += 1
    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())
    #pkl.dump(word_freqs,open('word_freq_dev.pkl','w'))
    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]
    worddict = OrderedDict()
    worddict['_PAD_'] = 0 # default, padding
    worddict['_UNK_'] = 1 # out-of-vocabulary
    worddict['_BOS_'] = 2 # begin of sentence token
    worddict['_EOS_'] = 3 # end of sentence token
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 4
        if word_freqs[ww] == cut_num:
            break

    joblib.dump(worddict, dst_path)
    print('Dict size', len(worddict))
    print('Done')




if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing snli_1.0 dataset')
    print('=' * 80)
    make_csv('../data/bs_new.utf8')
    #build_dictionary(['../train_jieba/shuffle.utf8.w2v'],'../train_jieba/jieba-enli.dict',4)
    #split_query_title_label('../data/pa.utf8')

