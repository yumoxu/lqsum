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
import ir.ir_config as ir_config

if config.grain != 'para':
    raise ValueError('Set grain to para before using this module!')

def _rank(cid, query):
    res = tfidf_tools.build_rel_scores_tf_para(cid, query)
    rel_scores = res['rel_scores']
    processed_paras = res['processed_paras']
    original_paras = res['original_paras']

    # get sid2score
    sid2score = dict()
    abs_idx = 0
    for doc_idx, doc in enumerate(original_paras):
        for para_idx, para in enumerate(doc):
            pid = config.SEP.join((str(doc_idx), str(para_idx)))
            score = rel_scores[abs_idx]
            sid2score[pid] = score

            abs_idx += 1

    # rank scores
    sid_score_list = rank_sent.sort_sid2score(sid2score)
    # include sentences in records
    rank_records = rank_sent.get_rank_records(sid_score_list, sents=original_paras)
    # rank_records = rank_sent.get_rank_records(sid_score_list)

    return rank_records


def rank_e2e():
    """

    :param pool_func: avg, max, or None (for integrated query).
    :return:
    """
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF)
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
    ir_rec_dp = join(path_parser.summary_rank, ir_config.IR_RECORDS_DIR_NAME_TF)

    if exists(ir_rec_dp):
        raise ValueError('qa_rec_dp exists: {}'.format(ir_rec_dp))
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
        }

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(ir_rec_dp, cid), retrieved_items=retrieved_items)


if __name__ == '__main__':
    rank_e2e()
    # ir_rank2records()
