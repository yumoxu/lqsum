# -*- coding: utf-8 -*-
import sys
import io
from os import listdir
from os.path import join, dirname, abspath, exists

sys_path = dirname(abspath(__file__))
for _ in range(3):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

from data.dataset_parser import dataset_parser
import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_meta
import utils.tools as tools
from utils.query_tools import get_cid2masked_query, get_cid2raw_query


def build_test_cid_query_dicts(query_type, with_sub):
    """

    :return:
    """
    if query_type != config.NARR:
        raise NotImplementedError
    
    query_info = get_cid2masked_query(query_type, with_sub=with_sub)
    
    cids = tools.get_test_cc_ids()
    test_cid_query_dicts = []

    for cid in cids:
        query = tools.get_query_w_cid(query_info, cid=cid)

        print('query: {}'.format(query))
        test_cid_query_dicts.append({
            'cid': cid,
            'query': query,
        })

    return test_cid_query_dicts


def build_test_cid_raw_query_dicts(query_type):
    """

    :return:
    """
    if query_type != config.NARR:
        raise NotImplementedError
    
    query_info = get_cid2raw_query(query_type)
    
    cids = tools.get_test_cc_ids()
    test_cid_query_dicts = []

    for cid in cids:
        query = tools.get_query_w_cid(query_info, cid=cid)

        print('query: {}'.format(query))
        test_cid_query_dicts.append({
            'cid': cid,
            'query': query,
        })

    return test_cid_query_dicts


def build_eli5_cid_query_dicts(question_fp, proc=True):
    """
    :return:
    """
    lines = io.open(question_fp).readlines()
    cid_query_dicts = []

    items = config_meta['test_year'].split('-')
    mds_cc_ids = None
    if len(items) >= 3 and items[2] == 'mds':
        mds_cc_ids = io.open(path_parser.data_eli5_mds_cc_ids).read().split(',')
        logger.info(f'Loaded {len(mds_cc_ids)} mds_cc_ids!')
    
    for line in lines:
        cid, question = line.rstrip('\n').split('\t')
        if mds_cc_ids and cid not in mds_cc_ids:
            continue

        if proc:
            question = dataset_parser._proc_sent(question, rm_dialog=False, rm_stop=False, stem=True, rm_short=None)
            
        cid_query_dicts.append({
            'cid': cid,
            'query': question,
        })
    return cid_query_dicts


def build_tdqfs_cid_query_dicts(query_fp, proc=True):
    """
    :return:
    """
    assert config_meta['test_year'] == 'tdqfs'
    lines = io.open(query_fp).readlines()
    cid_query_dicts = []

    items = config_meta['test_year'].split('-')
    for line in lines:
        cid, dom, query = line.rstrip('\n').split('\t')
        if proc:
            query = dataset_parser._proc_sent(query, rm_dialog=False, rm_stop=False, stem=True, rm_short=None)
        # print('query: {}'.format(query))
        cid_query_dicts.append({
            'cid': cid,
            'query': query,
        })
    return cid_query_dicts


def build_tdqfs_oracle_test_cid_query_dicts(query_fp):
    def _get_ref(cid):
        REF_DP = path_parser.data_tdqfs_summary_targets
        fp = join(REF_DP, '{}_{}'.format(cid, 0))
        ref = ''
        # for fn in fns:
        lines = io.open(fp, encoding='utf-8').readlines()
        for line in lines:
            ref += line.rstrip('\n')

        return ref

    assert config_meta['test_year'] == 'tdqfs'
    lines = io.open(query_fp).readlines()
    cids = [line.rstrip('\n').split('\t')[0] for line in lines]
    test_cid_query_dicts = []
    for cid in cids:
        ref = _get_ref(cid)
        logger.info('cid {}: {}'.format(cid, ref))

        test_cid_query_dicts.append({
            'cid': cid,
            'query': ref,
        })
    return test_cid_query_dicts


def build_oracle_test_cid_query_dicts():
    def _get_ref(cid):
        REF_DP = join(path_parser.data_summary_targets, config.test_year)

        # fns = [join(REF_DP, fn) for fn in listdir(REF_DP) if fn.startswith(cid)]
        # if len(fns) != 4:
        #     raise ValueError('Invalid #refs: {}'.format(len(fns)))

        fp = join(REF_DP, '{}_{}'.format(cid, 1))
        ref = ''
        # for fn in fns:
        lines = io.open(fp, encoding='utf-8').readlines()
        for line in lines:
            ref += line.rstrip('\n')

        return ref

    test_cid_query_dicts = []
    cids = tools.get_test_cc_ids()

    for cid in cids:
        ref = _get_ref(cid)
        logger.info('cid {}: {}'.format(cid, ref))

        test_cid_query_dicts.append({
            'cid': cid,
            'query': ref,
        })

    return test_cid_query_dicts


def build_test_question_info(tokenize_question):
    """

    :param tokenize_question: bool
    :return:
    """
    question_info = dict()
    for year in config.years:
        params = {
            'year': year,
            'tokenize_question': tokenize_question,
        }

        annual_question_info = dataset_parser.build_question_info(**params)
        question_info = {
            **annual_question_info,
            **question_info,
        }

    cids = tools.get_test_cc_ids()
    test_question_info = {}
    for cid in cids:
        test_question_info[cid] = question_info[cid]

    return test_question_info


if __name__ == '__main__':
    build_test_cid_query_dicts(tokenize_narr=None, concat_title_narr=True)