from tqdm import tqdm

from icecream import install
install()

import sys
import io

if 'cnndm' in sys.argv[6]:
    testset = 'cnndm'
    test_size = 11490
elif 'wikiref' in sys.argv[6]:
    testset = 'wikiref'
    test_size = 12000
else:
    raise ValueError(f'Test set unrecognized from the path: {sys.argv[6]}')

print(f'testset: {testset}')

REMOVE_START_END = True
TAG_ONLY = True


def output_to_tag(output_file, tag_file):
    lines = io.open(output_file).readlines()
    with io.open(tag_file, mode='a') as tag_f:
        for line in lines:
            items = line.strip('\n').split()
            if items[0].startswith('<s>'):
                items = items[1:-1]
            tags = [item[-1] for item in items]
            rec = ' '.join(tags)
            tag_f.write(rec + '\n')
    

def eval_tags(pred_tag_file, gold_tag_file):
    pred_items = [line.strip('\n').split() for line in io.open(pred_tag_file)]
    gold_items = [line.strip('\n').split() for line in io.open(gold_tag_file)]

    for pred_item, gold_itme in zip(pred_items, gold_items):
        pass
