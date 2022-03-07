import io
import re
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from tqdm import tqdm
from os.path import isfile, isdir, join, dirname, abspath
import sys
import nltk

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from data.dataset_parser import dataset_parser
import utils.config_loader as config
from utils.config_loader import logger, path_parser
import utils.tools as tools
import itertools

word_tokenizer = nltk.tokenize.word_tokenize


def print_dist(data_list, stat_var, end, higher_var=None, n_bins=10):
    if not higher_var:
        if stat_var == 'sent':
            higher_var = 'doc'
        elif stat_var == 'word':
            higher_var = 'sent'
        else:
            raise ValueError('Invalid stat_var: {}'.format(stat_var))

    data = np.array(data_list)
    total = float(len(data_list))
    logger.info('total #{0}: {1}'.format(higher_var, total))
    hist, bins = np.histogram(data, bins=np.linspace(0, end, n_bins, endpoint=False, dtype=np.int32))
    ratios = hist / total

    intervals = ['{0}-{1}'.format(bins[i], bins[i + 1]) for i in range(len(hist))]

    for i, h, r in zip(intervals, hist, ratios):
        logger.info('#{0} whose #{1} belongs to {2}: {3}, {4:2f}%'.format(higher_var, stat_var, i, h, r * 100))

    left = (1 - sum(ratios)) * 100
    logger.info(
        '#{0} whose #{1} belongs to >{2}: {3}, {4:2f}%'.format(higher_var, stat_var, bins[-1], total - sum(hist),
                                                               left))

    ratios = [str(r * 100) for r in ratios.tolist()]
    ratios.append(str(left))
    logger.info('\t'.join(ratios))


def print_nw_in_cluster(year):
    if year:
        years = [year]
    else:
        years = config.years

    for year in years:
        cc_ids = tools.get_cc_ids(year, model_mode='test')
        cid2nw = {}

        for cid in tqdm(cc_ids):
            original_sents, _ = dataset_parser.cid2sents(cid)
            nw = sum([len(word_tokenizer(sent)) for doc_sents in original_sents for sent in doc_sents])
            cid2nw[cid] = nw

        logger.info('Year: {}'.format(year))
        print_dist(list(cid2nw.values()), stat_var='word', end=10000, higher_var='cluster', n_bins=10)

        records = ['\t'.join((k, str(v))) for k, v in cid2nw.items()]
        print('===============cid2nw===============')
        print('\n'.join(records))

if __name__ == '__main__':
    print_nw_in_cluster(year='2007')
