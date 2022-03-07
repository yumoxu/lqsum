# -*- coding: utf-8 -*-
import sys
import os
import io
from os.path import join, dirname, abspath, exists

sys_path = dirname(abspath(__file__))
for _ in range(3):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

from tqdm import tqdm
import utils.config_loader as config
from utils.config_loader import logger, config_meta, path_parser
import utils.graph_io as graph_io
import utils.graph_tools as graph_tools
import utils.tfidf_tools as tfidf_tools
import utils.general_tools as general_tools
import utils.vec_tools as vec_tools
from utils.tools import get_text_dp_for_eli5

import summ.select_sent as select_sent
import summ.compute_rouge as rouge
from data.dataset_parser import dataset_parser
from multiprocessing import Pool


assert config_meta['test_year'] == 'tdqfsdoc' or config_meta['test_year'].split('_')[0] in ('wikiref', 'debatepedia', 'ducdoc', 'wikicat')

MODEL_NAME = f"grsum-{config_meta['test_year']}"
DIVERSITY_PARAM_TUPLE = (0, 'wan')
COS_THRESHOLD = 0.6
DAMP = 0.85
RM_DIALOG = False

# LENGTH_BUDGET_TUPLE = ('nw', 100)
LENGTH_BUDGET_TUPLE = ('ns', 3)

# for baseline in wikicat: lexrank
is_wikicat = config_meta['test_year'].split('_')[0] == 'wikicat'
if is_wikicat:
    DAMP = 1.0
    test_cid_query_dicts = [{'uid':uid, 'q':'THIS IS A QUERY PLACEHOLDER'} for uid in dataset_parser.uids]
else:
    test_cid_query_dicts = [{'uid':uid, 'q':dataset_parser.uid2instc[uid]['q']} for uid in dataset_parser.uids]


def _build_components(uid, q):
    sim_items = tfidf_tools.build_sim_items_e2e_tfidf_with_lexrank(uid, q, rm_dialog=RM_DIALOG)

    sim_mat = vec_tools.norm_sim_mat(sim_mat=sim_items['doc_sim_mat'], max_min_scale=False)
    rel_vec = vec_tools.norm_rel_scores(rel_scores=sim_items['rel_scores'], max_min_scale=False)

    assert len(rel_vec) == len(sim_mat), f'Incompatible sim_mat size: {sim_mat.shape} and rel_vec size: {rel_vec.shape} for uid: {uid}'

    processed_sents = sim_items['processed_sents']
    sid2abs = {}
    sid_abs = 0
    for doc_idx, doc in enumerate(processed_sents):
        for sent_idx, sent in enumerate(doc):
            sid = config.SEP.join((str(doc_idx), str(sent_idx)))
            sid2abs[sid] = sid_abs
            sid_abs += 1

    components = {
        'sim_mat': sim_mat,
        'rel_vec': rel_vec,
        'sid2abs': sid2abs,
    }

    return components


def _build_components_e2e_mp(params):
    sim_mat_dp = params.pop('sim_mat_dp')
    rel_vec_dp = params.pop('rel_vec_dp')
    sid2abs_dp = params.pop('sid2abs_dp')

    components = _build_components(**params)

    graph_io.dump_sim_mat(sim_mat=components['sim_mat'], sim_mat_dp=sim_mat_dp, cid=params['uid'])
    graph_io.dump_rel_vec(rel_vec=components['rel_vec'], rel_vec_dp=rel_vec_dp, cid=params['uid'])
    graph_io.dump_sid2abs(sid2abs=components['sid2abs'], sid2abs_dp=sid2abs_dp, cid=params['uid'])


def build_components_e2e_mp():
    dp_params = {
        'model_name': MODEL_NAME,
        'n_iter': None,
        'mode': 'w',
    }

    summ_comp_root = graph_io.get_summ_comp_root(**dp_params)
    sim_mat_dp = graph_io.get_sim_mat_dp(summ_comp_root, mode='w')
    rel_vec_dp = graph_io.get_rel_vec_dp(summ_comp_root, mode='w')
    sid2abs_dp = graph_io.get_sid2abs_dp(summ_comp_root, mode='w')

    for dict in test_cid_query_dicts:
        dict['sim_mat_dp'] = sim_mat_dp
        dict['rel_vec_dp'] = rel_vec_dp
        dict['sid2abs_dp'] = sid2abs_dp

    p = Pool(30)
    p.map(_build_components_e2e_mp, test_cid_query_dicts)


def build_components_e2e():
    dp_params = {
        'model_name': MODEL_NAME,
        'n_iter': None,
        'mode': 'w',
    }

    summ_comp_root = graph_io.get_summ_comp_root(**dp_params)
    sim_mat_dp = graph_io.get_sim_mat_dp(summ_comp_root, mode='w')
    rel_vec_dp = graph_io.get_rel_vec_dp(summ_comp_root, mode='w')
    sid2abs_dp = graph_io.get_sid2abs_dp(summ_comp_root, mode='w')

    for params in tqdm(test_cid_query_dicts):
        components = _build_components(**params)
        graph_io.dump_sim_mat(sim_mat=components['sim_mat'], sim_mat_dp=sim_mat_dp, cid=params['uid'])
        graph_io.dump_rel_vec(rel_vec=components['rel_vec'], rel_vec_dp=rel_vec_dp, cid=params['uid'])
        graph_io.dump_sid2abs(sid2abs=components['sid2abs'], sid2abs_dp=sid2abs_dp, cid=params['uid'])


def score_e2e():
    if DAMP == 1.0:
        damp = 0.85
        use_rel_vec = False
    else:
        damp = DAMP
        use_rel_vec = True

    graph_tools.score_end2end(model_name=MODEL_NAME,
                              damp=damp,
                              use_rel_vec=use_rel_vec,
                              cc_ids=dataset_parser.uids)


def rank_e2e():
    graph_tools.rank_end2end(model_name=MODEL_NAME,
                             diversity_param_tuple=DIVERSITY_PARAM_TUPLE,
                             retrieved_dp=None,
                             cc_ids=dataset_parser.uids,
                             rm_dialog=RM_DIALOG)


def select_e2e():
    params = {
        'model_name': MODEL_NAME,
        'diversity_param_tuple': DIVERSITY_PARAM_TUPLE,
        'length_budget_tuple': LENGTH_BUDGET_TUPLE,
        'cos_threshold': COS_THRESHOLD,
        'rm_dialog': RM_DIALOG,
        'cc_ids': dataset_parser.uids,
    }
    select_sent.select_end2end_for_eli5(**params)


def select2guidance():
    params = {
        'model_name': MODEL_NAME,
        'cos_threshold': COS_THRESHOLD,
        'diversity_param_tuple': DIVERSITY_PARAM_TUPLE,
        'length_budget_tuple': LENGTH_BUDGET_TUPLE,
    }
    text_dp = get_text_dp_for_eli5(**params)
    from utils.tools import get_guidance_fp_for_eli5
    guidance_fp = get_guidance_fp_for_eli5(**params)

    with io.open(guidance_fp, 'a') as guidance_f:
        for uid in tqdm(dataset_parser.uids):
            lines = io.open(text_dp/uid).readlines()
            lines = [line.strip('\n') for line in lines]
            guidance = ' '.join(lines)
            guidance_f.write(guidance+'\n')
    logger.info(f'Guidance has been saved to: {guidance_fp}')


def compute_rouge():
    text_params = {
        'model_name': MODEL_NAME,
        'diversity_param_tuple': DIVERSITY_PARAM_TUPLE,
        'length_budget_tuple': LENGTH_BUDGET_TUPLE,
        'cos_threshold': COS_THRESHOLD,
    }

    text_dp = get_text_dp_for_eli5(**text_params)

    rouge_parmas = {
        'text_dp': text_dp,
        'ref_dp': None,  # FIXME generate ref files
    }
    if LENGTH_BUDGET_TUPLE[0] == 'nw':
        rouge_parmas['length'] = LENGTH_BUDGET_TUPLE[1]
    
    output = rouge.compute_rouge_for_tdqfs(**rouge_parmas)
    return output
    

if __name__ == '__main__':
    # build_components_e2e()
    build_components_e2e_mp()
    score_e2e()
    rank_e2e()
    select_e2e()
    select2guidance()
    # compute_rouge()
