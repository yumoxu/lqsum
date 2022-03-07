import logging
import logging.config
import yaml
from io import open
import os
from os.path import join, dirname, abspath
import warnings
import sys
from pytorch_transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, BertForQuestionAnswering
import torch
from pathlib import Path

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def deprecated(func):
    """
        This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.
    """

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


class PathParser:
    def __init__(self, path_type):
        self.remote_root = Path('/disk/nfs/ostrom/s1617290')
        self.proj_name = 'lqsum'

        if path_type == 'local':
            self.proj_root = Path(f'/Users/KevinXU/programming/git_yumoxu/{self.proj_name}')
        elif path_type == 'afs':
            self.proj_root = self.remote_root / self.proj_name
        else:
            raise ValueError(f'Invalid path_type: {path_type}')

        print(f'Set proj_root to: {self.proj_root}')
        
        self.performances = self.proj_root / 'performances'
        self.log = self.proj_root / 'log'
        self.data = self.proj_root / 'data'
        self.bart_out = self.proj_root / 'bart_out'

        # cnndm
        self.cnndm_raw = self.data / 'cnndm' / 'raw'
        self.cnndm_url = self.cnndm_raw / 'url_lists'
        self.cnndm_raw_cnn_story = self.cnndm_raw / 'cnn' / 'stories'
        self.cnndm_raw_dm_story = self.cnndm_raw / 'dailymail' / 'stories'
        
        self.wikicat = self.data / 'wikicat'
        self.wikiref = self.data / 'wikiref'
        self.debatepedia = self.data / 'debatepedia'

        # duc
        self.duc = self.data / 'duc'
        self.duc_cluster =  self.duc / 'duc_cluster'
        self.duc_summary = self.duc / 'duc_summary'
        self.duc_query = self.duc / 'duc_query'
        
        self.raw_query = self.data / 'raw_query'
        self.parsed_query = self.data / 'parsed_query'
        self.masked_query = self.duc / 'masked_query'


        # tdqfs
        self.tdqfs = self.data / 'tdqfs'
        self.tdqfs_doc = self.tdqfs / 'docs'
        self.tdqfs_query = self.tdqfs / 'query_info.txt'
        self.tdqfs_summary = self.tdqfs / 'summaries'  # organized via clusters
        self.tdqfs_summary_targets = self.tdqfs / 'summary_targets'  # flat

        self.data_mn_summary_targets = self.data / 'multinews_rr' / 'test_mn_summary'
        
        # set res
        self.res = self.proj_root / 'res'

        self.model_save = self.proj_root / 'model'

        self.pred = self.proj_root / 'pred'

        self.summary_rank = self.proj_root / 'rank'
        self.summary_text = self.proj_root / 'text'
        self.guidance = self.proj_root / 'guidance'
        self.graph = self.proj_root / 'graph'

        self.graph_rel_scores = self.graph / 'rel_scores'  # for dumping relevance scores
        self.graph_token_logits = self.graph / 'token_logits'  # for dumping relevance scores

        self.rouge = self.proj_root / 'rouge'
        self.tune = self.proj_root / 'tune'
        self.cont_sel = self.proj_root / 'cont_sel'
        self.mturk = self.proj_root / 'mturk'

        self.afs_rouge_dir = self.remote_root / 'ROUGE-1.5.5' / 'data'
        self.local_rouge_dir = '/Users/KevinXU/Programming/git_yumoxu/pyrouge/RELEASE-1.5.5/data'


config_root = Path(os.path.dirname(os.path.dirname(__file__))) / 'config'

# meta
config_meta_fp = config_root / 'config_meta.yml'
config_meta = yaml.load(open(config_meta_fp, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
path_type = config_meta['path_type']
mode = config_meta['mode']
debug = config_meta['debug']
grain = config_meta['grain']

path_parser = PathParser(path_type=path_type)

# model
meta_model_name = config_meta['model_name']
test_year = config_meta['test_year']
model_name = 'lqsum'

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = join(path_parser.log, '{0}.log'.format(model_name))
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s", "%m/%d/%Y %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info(f'model name: {model_name}')

NARR = 'narr'
TITLE = 'title'
QUERY = 'query'
NONE = 'None'
HEADLINE = 'headline'
SEP = '_'
NEG_SEP = ';'

query_types = (NARR, HEADLINE)
years = ['2005', '2006', '2007']
