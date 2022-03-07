# -*- coding: utf-8 -*-
import sys

import os
from os.path import join, dirname, abspath, exists

sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import utils.config_loader as config
from utils.config_loader import logger, path_parser
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent
import utils.tools as tools
import tools.tfidf_tools as tfidf_tools
import tools.general_tools as general_tools
import ir.ir_tools as ir_tools

from tqdm import tqdm
import shutil
import ir.ir_config as ir_config
import summ.compute_rouge as rouge
import numpy as np


"""
    This module has been revised for getting all sentences in clusters, i.e., REMOVE_DIALOG==True.

    In ir_config, added REMOVE_DIALOG.

    in tfidf_tools.build_rel_scores_tf, added rm_dialog as parameter.
    
    In _rank, added rm_dialog as parameter.

    In rank_e2e, new build_test_cid_query_dicts() is used where we use only Query Narrative since this function is only built for Masked Query Narrative.
    
    (It doen't matter if the IR score is worse since we use only sentences.)
"""


if config.grain != 'sent':
    raise ValueError('Invalid grain: {}'.format(config.grain))


def _rank(cid, query, rm_dialog):
    res = tfidf_tools.build_rel_scores_tf(cid, query, rm_dialog=rm_dialog)  # todo: add
    rel_scores = res['rel_scores']
    processed_sents = res['processed_sents']
    original_sents = res['original_sents']

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
    """

    :param pool_func: avg, max, or None (for integrated query).
    :return:
    """
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF)

    if ir_config.QUERY_TYPE == 'REF':
        test_cid_query_dicts = general_tools.build_oracle_test_cid_query_dicts()
    else:
        test_cid_query_dicts = general_tools.build_test_cid_query_dicts(query_type='narr')

    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for cid_query_dict in tqdm(test_cid_query_dicts):
        params = {
            **cid_query_dict,
            'rm_dialog': ir_config.REMOVE_DIALOG,
        }
        rank_records = _rank(**params)
        rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, params['cid']), with_rank_idx=False)

    logger.info('[RANK SENT] successfully dumped rankings to: {}'.format(rank_dp))


def ir_rank2records():
    ir_rec_dp = join(path_parser.summary_rank, ir_config.IR_RECORDS_DIR_NAME_TF)

    if exists(ir_rec_dp):
        raise ValueError('ir_rec_dp exists: {}'.format(ir_rec_dp))
    os.mkdir(ir_rec_dp)

    cids = tools.get_test_cc_ids()
    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': ir_config.IR_MODEL_NAME_TF,
            'cid': cid,
            'filter_var': ir_config.FILTER_VAR,
            'filter': ir_config.FILTER,
            'deduplicate': ir_config.DEDUPLICATE,
            'min_ns': ir_config.IR_MIN_NS,
            'prune': True,
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

    cids = tools.get_test_cc_ids()
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
        performance = rouge.compute_rouge_for_cont_sel_in_sentences(ir_tune_dp)
        with open(ir_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if FILTER in ('conf', 'comp'):
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = '{0}\t{1}\n'.format(filter_var, performance)

            out_f.write(rec)

    if exists(ir_tune_dp):  # remove previous output
        shutil.rmtree(ir_tune_dp)


def compute_rouge_for_oracle():
    """
        The rec dp for oracle saves text for comparing against refecence.

    :return:
    """
    ir_rec_dp = join(path_parser.summary_rank, ir_config.IR_RECORDS_DIR_NAME_TF)

    if exists(ir_rec_dp):
        raise ValueError('ir_rec_dp exists: {}'.format(ir_rec_dp))
    os.mkdir(ir_rec_dp)

    cids = tools.get_test_cc_ids()
    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': ir_config.IR_MODEL_NAME_TF,
            'cid': cid,
            'filter_var': ir_config.FILTER_VAR,
            'filter': ir_config.FILTER,
            'deduplicate': ir_config.DEDUPLICATE,
            'min_ns': ir_config.IR_MIN_NS,
            'prune': True,
        }

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        summary = '\n'.join([item[-1] for item in retrieved_items])
        with open(join(ir_rec_dp, cid), mode='a', encoding='utf-8') as out_f:
            out_f.write(summary)

    performance = rouge.compute_rouge_for_ablation_study(ir_rec_dp)
    logger.info(performance)


def select_e2e():
    """
        This function is for ablation study: Retrieval Only.

    """
    params = {
        'model_name': ir_config.IR_MODEL_NAME_TF,
        'cos_threshold': 0.6,
        # 'retrieved_dp': join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF),
        'max_n_summary_words': 500,
        'rm_dialog': ir_config.REMOVE_DIALOG,
    }

    select_sent.select_end2end(**params)


def compute_rouge():
    text_params = {
        'model_name': ir_config.IR_MODEL_NAME_TF,
        'cos_threshold': 0.6,
    }
    output = rouge.compute_rouge(**text_params)


def eval_unilm_out():
    """
        Copied from rr/main.py.
    """
    from querysum.unilm_utils.unilm_eval import UniLMEval

    cids = tools.get_test_cc_ids()
    unilm_eval = UniLMEval(marge_config=ir_config, 
        pre_tokenize_sent=False, max_eval_len=250, cluster_ids=cids)
    unilm_eval.build_and_eval_unilm_out()


def build_unilm_input(src):
    """
        Copied from rr/main.py.
    """
    
    from querysum.unilm_utils.unilm_input import UniLMInput

    if src == 'rank':
        rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF)
        text_dp = None
    elif src == 'text':
        rank_dp = None
        text_dp = get_text_dp()
    
    cids = tools.get_test_cc_ids()
    unilm_input = UniLMInput(marge_config=ir_config,
        rank_dp=rank_dp,
        text_dp=text_dp, 
        fix_input=True, 
        multi_pass=ir_config.MULTI_PASS,
        cluster_ids=cids,
        prepend_len=ir_config.PREPEND_LEN)

    if src == 'rank':
        unilm_input.build_from_rank()
    elif src == 'text':
        unilm_input.build_from_text()


if __name__ == '__main__':
    rank_e2e()
    # ir_rank2records()
    # compute_rouge_for_oracle()
    tune()

    # build_unilm_input('rank')
    # eval_unilm_out()
    # select_e2e()
    # compute_rouge()
