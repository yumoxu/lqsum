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


def _rank(cid, query):
    sim_items = tfidf_tools.build_sim_items_e2e(cid, query, mask_intra=False)
    rel_scores = sim_items['rel_scores']
    processed_sents = sim_items['processed_sents']
    original_sents = sim_items['original_sents']

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
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TFIDF)

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


# for relevance estimation
# compression = 0.5
# cos_threshold = 1.0
# relv_sents_dn_base = 'ir_tfidf_2007-{}_comp'.format(compression)  # for saving text file
def ir_rank2records():
    ir_rec_dp = join(path_parser.summary_rank, ir_config.IR_RECORDS_DIR_NAME_TFIDF)

    if exists(ir_rec_dp):
        raise ValueError('qa_rec_dp exists: {}'.format(ir_rec_dp))
    os.mkdir(ir_rec_dp)

    cids = tools.get_test_cc_ids()
    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': ir_config.IR_MODEL_NAME_TFIDF,
            'cid': cid,
            'filter_var': ir_config.FILTER_VAR,
            'filter': ir_config.FILTER,
            'deduplicate': ir_config.DEDUPLICATE,
        }

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(ir_rec_dp, cid), retrieved_items=retrieved_items)


if __name__ == '__main__':
    # rank_e2e()
    ir_rank2records()
