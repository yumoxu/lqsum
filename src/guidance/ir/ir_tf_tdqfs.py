# -*- coding: utf-8 -*-
import sys

import os
from os import listdir
from os.path import join, dirname, abspath, exists

sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import utils.config_loader as config
from utils.config_loader import logger, path_parser
import summ.rank_sent as rank_sent
import utils.tools as tools
import tools.tfidf_tools as tfidf_tools
import tools.general_tools as general_tools
import ir.ir_tools as ir_tools

from tqdm import tqdm
import shutil
import ir.ir_config as ir_config
import summ.compute_rouge as rouge
import numpy as np
import io
from data.dataset_parser import dataset_parser

from multiprocessing import Pool
import summ.select_sent as select_sent
import itertools

assert config.grain == 'sent', f'Invalid grain: {config.grain}'
assert ir_config.test_year.startswith('tdqfs'), f'set ir_config.test_year to tdqfs! now: {ir_config.test_year}'

sentence_dp = path_parser.data_tdqfs_sentences
query_fp = path_parser.data_tdqfs_queries
summary_target_dp = path_parser.data_tdqfs_summary_targets

if ir_config.QUERY_TYPE == 'REF':
    test_cid_query_dicts = general_tools.build_tdqfs_oracle_test_cid_query_dicts(query_fp=query_fp)
else:
    test_cid_query_dicts = general_tools.build_tdqfs_cid_query_dicts(query_fp=query_fp, proc=True)

cids = [cq_dict['cid'] for cq_dict in test_cid_query_dicts]

def get_sentences_flat(cid):
    """
        Get sentences in a flat list (i.e., no document infomation).
    """
    cc_dp = join(sentence_dp, cid)
    fns = [fn for fn in listdir(cc_dp)]
    lines = itertools.chain(*[io.open(join(cc_dp, fn)).readlines() for fn in fns])
    sentences = [line.strip('\n') for line in lines]

    original_sents = []
    processed_sents = []
    for ss in sentences:
        ss_origin = dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
        ss_proc = dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)

        if ss_proc:  # make sure the sent is not removed, i.e., is not empty and is not in a dialog
            original_sents.append(ss_origin)
            processed_sents.append(ss_proc)
    
    return [original_sents], [processed_sents]


def get_sentences(cid):
    """
        Get sentences in a nested list (i.e., organized with documents).
    """
    cc_dp = join(sentence_dp, cid)
    original_sents, processed_sents = [], []
    for fn in listdir(cc_dp):
        lines = io.open(join(cc_dp, fn)).readlines()
        sentences = [line.strip('\n') for line in lines]

        origin, proc = [], []
        for ss in sentences:
            ss_origin = dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
            ss_proc = dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)

            if ss_proc:  # make sure the sent is not removed, i.e., is not empty and is not in a dialog
                origin.append(ss_origin)
                proc.append(ss_proc)

        original_sents.append(origin)
        processed_sents.append(proc)
    
    return original_sents, processed_sents


def _rank(cid, query):
    original_sents, processed_sents = get_sentences(cid)
    rel_scores = tfidf_tools._compute_rel_scores_tf_dot(processed_sents, query)

    # get sid2score
    sid2score = dict()
    abs_idx = 0
    for doc_idx, doc in enumerate(processed_sents):
        for sent_idx, sent in enumerate(doc):
            sid = config.SEP.join((str(doc_idx), str(sent_idx)))
            score = rel_scores[abs_idx]
            sid2score[sid] = score

            abs_idx += 1

    # rank scores
    sid_score_list = rank_sent.sort_sid2score(sid2score)
    # include sentences in records
    rank_records = rank_sent.get_rank_records(sid_score_list, sents=original_sents)
    # rank_records = rank_sent.get_rank_records(sid_score_list)

    return rank_records


def rank_e2e():
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF)
    assert not exists(rank_dp), f'rank_dp exists: {rank_dp}'
    os.mkdir(rank_dp)

    for cid_query_dict in tqdm(test_cid_query_dicts):
        rank_records = _rank(**cid_query_dict)
        rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, cid_query_dict['cid']), with_rank_idx=False)

    logger.info('Successfully dumped rankings to: {}'.format(rank_dp))


def _rank_core(cq_dict):
    cid = cq_dict['cid']
    query = cq_dict['query']
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF)
    original_sents, processed_sents = get_sentences(cid)
    rel_scores = tfidf_tools._compute_rel_scores_tf_dot(processed_sents, query)

    # get sid2score
    sid2score = dict()
    abs_idx = 0
    for doc_idx, doc in enumerate(processed_sents):
        for sent_idx, sent in enumerate(doc):
            sid = config.SEP.join((str(doc_idx), str(sent_idx)))
            score = rel_scores[abs_idx]
            sid2score[sid] = score
            abs_idx += 1

    sid_score_list = rank_sent.sort_sid2score(sid2score)
    rank_records = rank_sent.get_rank_records(sid_score_list, sents=original_sents)
    rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, cid), with_rank_idx=False)


def rank_e2e_multiproc():
    p = Pool(20)
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF)

    assert not exists(rank_dp), f'rank_dp exists: {rank_dp}'
    os.mkdir(rank_dp)

    p.map(_rank_core, test_cid_query_dicts)


def ir_rank2records():
    ir_rec_dp = join(path_parser.summary_rank, ir_config.IR_RECORDS_DIR_NAME_TF)
    assert not exists(ir_rec_dp), f'ir_rec_dp exists: {ir_rec_dp}'
    os.mkdir(ir_rec_dp)

    # cids = tools.get_test_cc_ids()
    cids = [c_q_dict['cid'] for c_q_dict in test_cid_query_dicts]
    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': ir_config.IR_MODEL_NAME_TF,
            'cid': cid,
            'filter_var': ir_config.FILTER_VAR,
            'filter': ir_config.FILTER,
            'deduplicate': ir_config.DEDUPLICATE,
            'min_ns': ir_config.IR_MIN_NS,
            # 'prune': True,
            'prune': False,
        }

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(ir_rec_dp, cid), retrieved_items=retrieved_items)


def tune():
    """
        Tune IR confidence / compression rate based on Recall Rouge 2.
    :return:
    """
    FILTER = 'topK'
    if FILTER in ('conf', 'comp'):
        tune_range = np.arange(0.05, 1.05, 0.05)
    else:
        interval = 10
        tune_range = range(interval, 200+interval, interval)

    ir_tune_dp = join(path_parser.summary_rank, ir_config.IR_TUNE_DIR_NAME_TF)
    ir_tune_result_fp = join(path_parser.tune, ir_config.IR_TUNE_DIR_NAME_TF)
    with open(ir_tune_result_fp, mode='a', encoding='utf-8') as out_f:
        headline = 'Filter\tRecall\tF1\n'
        out_f.write(headline)

    # cids = tools.get_test_cc_ids()
    for filter_var in tune_range:
        if exists(ir_tune_dp):  # remove previous output
            shutil.rmtree(ir_tune_dp)
        os.mkdir(ir_tune_dp)

        for cid in tqdm(cids):
            retrieval_params = {
                'model_name': ir_config.IR_MODEL_NAME_TF,
                'cid': cid,
                'filter_var': filter_var,
                'filter': FILTER,
                'deduplicate': None,  # ir_config.DEDUPLICATE
                'min_ns': ir_config.IR_MIN_NS,
                # 'prune': True,
            }
            retrieved_items = ir_tools.retrieve(**retrieval_params)

            summary = '\n'.join([item[-1] for item in retrieved_items])
            # print(summary)
            with open(join(ir_tune_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(summary)

        # performance = rouge.compute_rouge_for_dev(ir_tune_dp, tune_centrality=False)
        # performance = rouge.compute_rouge_for_cont_sel_in_sentences(ir_tune_dp)
        performance = rouge.compute_rouge_for_cont_sel_in_sentences_tdqfs(text_dp=ir_tune_dp, ref_dp=summary_target_dp)
        with open(ir_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if FILTER in ('conf', 'comp'):
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = '{0}\t{1}\n'.format(filter_var, performance)

            out_f.write(rec)

    if exists(ir_tune_dp):  # remove previous output
        shutil.rmtree(ir_tune_dp)


def select_e2e():
    params = {
        'model_name': ir_config.IR_MODEL_NAME_TF,
        'length_budget_tuple': ('nw', 250),
        # 'cos_threshold': 1.0,
        'cos_threshold': 0.6,
        'retrieved_dp': join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF),
        'cc_ids': cids,
    }
    select_sent.select_end2end_for_eli5(**params)


def compute_rouge():
    text_params = {
        'model_name': ir_config.IR_MODEL_NAME_TF,
        'length_budget_tuple': ('nw', 250),
        'cos_threshold': 0.6,  # 1.0
    }

    text_dp = tools.get_text_dp_for_eli5(**text_params)

    rouge_parmas = {
        'text_dp': text_dp,
        'ref_dp': summary_target_dp,
        'length': 250
    }
    
    output = rouge.compute_rouge_for_tdqfs(**rouge_parmas)
    return output


if __name__ == '__main__':
    # rank_e2e_multiproc()
    # ir_rank2records()

    # compute_rouge_for_oracle()
    tune()

    # select_e2e()
    # compute_rouge()
