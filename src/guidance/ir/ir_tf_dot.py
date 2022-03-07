# -*- coding: utf-8 -*-
import sys

import os
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

"""
    Module for IR with TF Dot.
"""

def _rank(cid, query):
    res = tfidf_tools.build_rel_scores_tf_dot(cid, query)
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
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF_DOT)
    test_cid_query_dicts = general_tools.build_test_cid_query_dicts(tokenize_narr=False,
                                                                    concat_title_narr=True)

    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for cid_query_dict in tqdm(test_cid_query_dicts):
        params = {
            **cid_query_dict,
        }
        rank_records = _rank(**params)
        rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, params['cid']), with_rank_idx=False)

    logger.info('[RANK SENT] successfully dumped rankings to: {}'.format(rank_dp))


def ir_rank2records():
    ir_rec_dp = join(path_parser.summary_rank, ir_config.IR_RECORDS_DIR_NAME_TF_DOT)

    if exists(ir_rec_dp):
        raise ValueError('qa_rec_dp exists: {}'.format(ir_rec_dp))
    os.mkdir(ir_rec_dp)

    cids = tools.get_test_cc_ids()
    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': ir_config.IR_MODEL_NAME_TF_DOT,
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
    if ir_config.FILTER in ('conf', 'comp'):
        tune_range = np.arange(0.05, 1.05, 0.05)
    else:
        tune_range = range(25, 525, 25)

    ir_tune_dp = join(path_parser.summary_rank, ir_config.IR_TUNE_DIR_NAME_TF_DOT)
    ir_tune_result_fp = join(path_parser.tune, ir_config.IR_TUNE_DIR_NAME_TF_DOT)
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
                'model_name': ir_config.IR_MODEL_NAME_TF_DOT,
                'cid': cid,
                'filter_var': filter_var,
                'filter': ir_config.FILTER,
                'deduplicate': ir_config.DEDUPLICATE,
                'min_ns': ir_config.IR_MIN_NS,
                'prune': True,
            }

            retrieved_items = ir_tools.retrieve(**retrieval_params)

            summary = '\n'.join([item[-1] for item in retrieved_items])
            # print(summary)
            with open(join(ir_tune_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(summary)

        performance = rouge.compute_rouge_for_dev(ir_tune_dp, tune_centrality=False)
        with open(ir_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if ir_config.FILTER in ('conf', 'comp'):
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = '{0}\t{1}\n'.format(filter_var, performance)

            out_f.write(rec)


if __name__ == '__main__':
    rank_e2e()
    ir_rank2records()
    # tune()
