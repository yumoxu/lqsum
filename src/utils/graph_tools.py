from os.path import exists, join, dirname, abspath
import itertools
import sys
import copy
import sklearn
import os
import io
from tqdm import tqdm
from lexrank import STOPWORDS, LexRank

import utils.tools as tools
import utils.graph_io as graph_io
import utils.config_loader as config
from utils.config_loader import logger
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent
from data.dataset_parser import dataset_parser
from ir.ir_tools import load_retrieved_sentences

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def _score_graph_initially(sim_mat, rel_vec, cid, damp, abs2sid=None, rm_dialog=True):
    doc_place_holder = [['test sentence 1', 'test sentence 2'], ['test sentence 3']]
    lxr = LexRank(doc_place_holder, stopwords=STOPWORDS['en'])
    params = {
        'similarity_matrix': sim_mat,
        'threshold': None,
        'fast_power_method': True,
        'rel_vec': rel_vec,
        'damp': damp,
    }
    scores = lxr.rank_sentences_with_sim_mat(**params)

    sid2score = dict()

    for abs, sc in enumerate(scores):
        sid2score[abs2sid[abs]] = sc

    return sid2score


def _rank_with_diversity_penalty_wan(sid2score, sid2abs, sim_mat, omega=10, original_sents=None):
    """

    :param sid2score:
    :param sid2abs:
    :param sim_mat:
    :param omega:
    :param original_sents:
    :param alg: diversity algorithm; wan.

    :return:
    """
    if omega < 0:
        raise ValueError('Invalid omega: {}'.format(omega))

    # norm sim_mat
    sim_mat_normed = sklearn.preprocessing.normalize(sim_mat, axis=1, norm='l1')
    sid_score_list_selected = []
    # n_iter = 0

    sid2score_ar = copy.deepcopy(sid2score)

    # while sid2score_ar and n_iter <= max_n_iter:
    while sid2score_ar:
        sid_score_list = rank_sent.sort_sid2score(sid2score=sid2score_ar)
        sid_0, _ = sid_score_list[0]
        sid_score_list_selected.append(sid_score_list[0])

        ii = sid2abs[sid_0]
        del sid2score_ar[sid_0]
        for sid_j in sid2score_ar:  # penalize remaining sentences
            jj = sid2abs[sid_j]
            info_rich_ii = sid2score[sid_0]
            penalty = omega * sim_mat_normed[jj, ii] * info_rich_ii

            sid2score_ar[sid_j] -= penalty

    rank_records = rank_sent.get_rank_records(sid_score_list=sid_score_list_selected, sents=original_sents)
    return rank_records


def score_end2end(model_name, n_iter=None, damp=0.85, use_rel_vec=True, rm_dialog=True, cc_ids=None):
    """
        cc_ids: set to None. Specify it for MN dataset.
    """
    dp_mode = 'r'
    dp_params = {
        'model_name': model_name,  # one model has only one suit of summary components but different ranking sys
        'n_iter': n_iter,
        'mode': dp_mode,
    }

    summ_comp_root = graph_io.get_summ_comp_root(**dp_params)
    sim_mat_dp = graph_io.get_sim_mat_dp(summ_comp_root, mode=dp_mode)
    rel_vec_dp = graph_io.get_rel_vec_dp(summ_comp_root, mode=dp_mode)
    sid2abs_dp = graph_io.get_sid2abs_dp(summ_comp_root, mode=dp_mode)

    sid2score_dp = graph_io.get_sid2score_dp(summ_comp_root, mode='w')

    dps = {
        'sim_mat_dp': sim_mat_dp,
        'rel_vec_dp': rel_vec_dp,
        'sid2abs_dp': sid2abs_dp,
    }

    if not cc_ids:
        cc_ids = tools.get_test_cc_ids()

    count = 0
    for cid in tqdm(cc_ids):
        comp_params = {
            **dps,
            'cid': cid,
        }
        components = graph_io.load_components(**comp_params)

        abs2sid = {}
        for sid, abs in components['sid2abs'].items():
            abs2sid[abs] = sid

        scoring_params = {
            'sim_mat': components['sim_mat'],
            'rel_vec': components['rel_vec'].transpose() if use_rel_vec else None,
            'cid': cid,
            'damp': damp,
            'abs2sid': abs2sid,
            'rm_dialog': rm_dialog,
        }

        sid2score = _score_graph_initially(**scoring_params)
        # if count % 100 ==0:
        #     logger.info(f"sim_mat: {components['sim_mat']}")
        #     logger.info(f"rel_vec: {components['rel_vec']}")
        #     logger.info(f"sid2score: {sid2score}")

        graph_io.dump_sid2score(sid2score=sid2score, sid2score_dp=sid2score_dp, cid=cid)

    logger.info('Scores have been dumped to: {}'.format(sid2score_dp))


def rank_end2end(model_name,
                 diversity_param_tuple,
                 component_name=None,
                 n_iter=None,
                 rank_dp=None,
                 retrieved_dp=None,
                 cc_ids=None,
                 rm_dialog=True):
    """

    :param model_name:
    :param diversity_param_tuple:
    :param component_name:
    :param n_iter:
    :param rank_dp:
    :param retrieved_dp:
    :param rm_dialog: only useful when retrieved_dp=None
    :return:
    """
    dp_mode = 'r'
    dp_params = {
        'n_iter': n_iter,
        'mode': dp_mode,
    }

    diversity_weight, diversity_algorithm = diversity_param_tuple
    if component_name:
        dp_params['model_name'] = component_name
    else:
        dp_params['model_name'] = model_name

    summ_comp_root = graph_io.get_summ_comp_root(**dp_params)
    sim_mat_dp = graph_io.get_sim_mat_dp(summ_comp_root, mode=dp_mode)
    rel_vec_dp = graph_io.get_rel_vec_dp(summ_comp_root, mode=dp_mode)
    sid2abs_dp = graph_io.get_sid2abs_dp(summ_comp_root, mode=dp_mode)
    sid2score_dp = graph_io.get_sid2score_dp(summ_comp_root, mode=dp_mode)

    if not rank_dp:
        rank_dp_params = {
            'model_name': model_name,
            'n_iter': n_iter,
            'diversity_param_tuple': diversity_param_tuple,
        }

        rank_dp = tools.get_rank_dp(**rank_dp_params)

    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    dps = {
        'sim_mat_dp': sim_mat_dp,
        'rel_vec_dp': rel_vec_dp,
        'sid2abs_dp': sid2abs_dp,
    }

    for cid in cc_ids:
        comp_params = {
            **dps,
            'cid': cid,
        }
        components = graph_io.load_components(**comp_params)
        sid2score = graph_io.load_sid2score(sid2score_dp, cid)

        if retrieved_dp:
            original_sents, _ = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)
        else:
            original_sents, _ = dataset_parser.cid2sents(cid, rm_dialog=rm_dialog)  # 2d lists, docs => sents

            # if config.test_year in ('wikiref', 'debatepedia'):
                # original_sents = inst['d']
            # else: # FIXME
                # NotImplementedError(f'Check implementation in ShiftSum codebase for test year: {config.test_year}')

        diversity_params = {
            'sid2score': sid2score,
            'sid2abs': components['sid2abs'],
            'sim_mat': components['sim_mat'],
            'original_sents': original_sents,
        }
        diversity_params['omega'] = diversity_weight
        rank_records = _rank_with_diversity_penalty_wan(**diversity_params)

        logger.info(f'cid: {cid}, #rank_records: {len(rank_records)}')
        rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, cid), with_rank_idx=False)

    logger.info('[GRAPH RANK] Finished. Rankings were dumped to: {}'.format(rank_dp))


def select_end2end(model_name, n_iter=None, omega=10, save_out_fp=None):
    params = {
        'model_name': model_name,
        'n_iter': n_iter,
        'cos_threshold': 1.0,  # do not pos cosine similarity criterion
        'omega': omega,
    }
    output = select_sent.select_end2end(**params)
    if save_out_fp:
        content = '\t'.join((str(omega), output))
        with io.open(save_out_fp, encoding='utf-8', mode='a') as f:
            f.write(content + '\n')
