# -*- coding: utf-8 -*-
import jieba
import codecs
from gensim import corpora
from gensim.summarization import bm25
import os
import re
import json
import multiprocessing as mp
import time
from functools import partial

"""
需要以下几个东西：

1. Query list and Query list index to QueryID and vice versa
2. Document list and Document list index to DocumentID and vice versa
3. 完成 bm25 之后，
4.
"""


def getQList_Qmap(input_json):
    """
    :param input_json:
    :return: query list, Listindex2ID and ID2Listindex
    """
    query_list = []
    ql_ind2id = {}
    id2ql_ind = {}

    instances = input_json['questions']
    for ind, instance in enumerate(instances):
        context = instance['context']
        ID = instance['ID']
        query_list.append(context)
        ql_ind2id[ind] = ID
        id2ql_ind[ID] = ind

    return query_list, ql_ind2id, id2ql_ind


def getDlist_Dmap(input_json):
    """
    :param input_json:
    :return: doc list, Listindex2ID and ID2Listindex
    """
    doc_list = []
    dl_ind2id = {}
    id2dl_ind = {}

    instances = input_json['document']
    for ind, instance in enumerate(instances):
        context = instance['context']
        ID = instance['ID']
        doc_list.append(context)
        dl_ind2id[ind] = ID
        id2dl_ind[ID] = ind

    return doc_list, dl_ind2id, id2dl_ind


def split_str(input_str, stop_word_l):
    split_words = jieba.cut(input_str, HMM=True)
    res = list(filter(lambda x: x not in stop_word_l, split_words))
    return res


def tokenization(input_l, stop_word_l):
    res = [split_str(input_str, stop_word_l) for input_str in input_l]
    return res


def get_bm_score(q, bm_model):
    return bm_model.get_scores(q)


def get_score_list(bm_model, query_list, multi_proc):
    if multi_proc:
        partial_work = partial(get_bm_score, bm_model=bm_model)
        pool = mp.Pool()
        scores = pool.map(partial_work, query_list)
    else:
        scores = []
        for query in query_list:
            scores.append(get_bm_score(query, bm_model))
    return scores


def evaluate_res(evaluation_map):
    total_n = len(evaluation_map)
    evaluation_iters = evaluation_map.items()
    print(total_n)
    print(evaluation_iters)
    corrects = sum(any(ele in k for ele in v) for k, v in evaluation_iters)
    acc = corrects/total_n
    return acc


# if __name__ == '__main__':
#     # get stop words from pretrained file
#     stop_word_file = 'stop_words.txt'
#     stop_word_l = []
#     with codecs.open(stop_word_file, 'r', 'utf-8') as f:
#         for line in f:
#             stop_word_l.append(line.rstrip('\n'))
#
#     input_file = os.path.join(os.getcwd(), 'tf_idf_cluster.json')
#     with codecs.open(input_file, 'r', 'utf-8') as file:
#         input_json = json.load(file)
#
#     # get Q_list, D_list and their corresponding mappings
#     Q_list, Qindex2ID, ID2Qindex = getQList_Qmap(input_json)
#     D_list, Dindex2ID, ID2Dindex = getDlist_Dmap(input_json)
#
#     # Tokenization the lists
#     Q_list = tokenization(Q_list, stop_word_l)
#     D_list = tokenization(D_list, stop_word_l)
#     # print(D_list[:5])
#     # print(Q_list[:5])
#
#     # get document dictionary
#     dictionary = corpora.Dictionary(D_list)
#     # print(len(dictionary))
#
#     # bm25 calculation
#     bm25Model = bm25.BM25(D_list)
#     sample_questions = Q_list[:]
#
#     # multiprocessing
#     start_time = time.time()
#     scores = get_score_list(bm25Model, sample_questions, multi_proc=True)
#     end_time = time.time()
#     time_used = (end_time - start_time)
#
#     # evaluation result
#     # evaluation_map should be like {'TRAIN_186_QUERY_0': ['TRAIN_186', 'TRAIN_1278', 'TRAIN_892']}
#     evaluation_map = {}
#     for ind, score in enumerate(scores):
#         query_ID = Qindex2ID[ind]
#         doc_scores = [(i, val) for i, val in enumerate(score)]
#         doc_scores.sort(key=lambda x: x[1], reverse=True)
#         doc_best_ids = [ele[0] for ele in doc_scores][:3]
#         evaluation_map[query_ID] = [Dindex2ID[ele] for ele in doc_best_ids]
#
#     suning_q_IDs = ['DEV_s001_QUERY_0','DEV_s001_QUERY_1','DEV_s001_QUERY_2','DEV_s001_QUERY_3', 'DEV_s001_QUERY_4',
#                     'DEV_s002_QUERY_0','DEV_s002_QUERY_1','DEV_s002_QUERY_2','DEV_s002_QUERY_3',
#                     'DEV_s003_QUERY_0','DEV_s003_QUERY_1','DEV_s003_QUERY_2','DEV_s003_QUERY_3']
#     print({id: evaluation_map[id] for id in suning_q_IDs})
#
#     # Start evaluation how the choice evaluation
#     accuracy = evaluate_res(evaluation_map)
#     print(accuracy)
