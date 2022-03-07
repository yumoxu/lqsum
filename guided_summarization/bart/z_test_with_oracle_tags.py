import torch
from fairseq.models.bart import (
    GuidedBARTModel, 
    GuidedBARTModelwithTagging, 
    GuidedBARTModelwithTaggingAndStdDec,
    GuidedBARTModelwithTaggingAndStdEncDec
)

import sys
from tqdm import tqdm
import logging

from icecream import install
install()

model_class = GuidedBARTModelwithTagging  # GuidedBARTModel, GuidedBARTModelwithTagging, GuidedBARTModelwithTaggingAndStdDec, GuidedBARTModelwithTaggingAndStdEncDec

bart = model_class.from_pretrained(
    sys.argv[4],
    checkpoint_file=sys.argv[5],
    data_name_or_path=sys.argv[6]
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 16

tag_mode = 'query_1.0'

TAG_MODE_VALUES=('', 'pre', 'pre-enc', 'pre_enc', 'post', 'post_enc', 'post-enc')
assert tag_mode in TAG_MODE_VALUES or 'query' in tag_mode, \
    f'Invalid tag_mode: {tag_mode}'
print(f'Generation with tagged guidance. Set tag_mode to {tag_mode}')

if 'cnndm' in sys.argv[6]:
    testset = 'cnndm'
    test_size = 11490
    min_len = 55
    max_len_b = 140
elif 'wikiref' in sys.argv[6]:
    testset = 'wikiref'
    test_size = 12000
    min_len = 35
    max_len_b = 90
else:
    raise ValueError(f'Test set unrecognized from the path: {sys.argv[6]}')

print(f'testset: {testset}, min_len: {min_len}, min_len_b: {max_len_b}')

with open(sys.argv[1]) as source, open(sys.argv[2]) as zs, open(sys.argv[3], 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    zline = zs.readline().strip()
    zlines = [zline]

    for sline, zline in tqdm(zip(source, zs), total=test_size):
        # if count <= 200:
            # count += 1
            # continue
        
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, zlines, 
                    tag_mode=tag_mode,
                    beam=4, lenpen=2.0, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=3, guided=True)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []
            zlines = []

        slines.append(sline.strip())
        zlines.append(zline.strip())
        count += 1
    
    if slines != []:
        hypotheses_batch = bart.sample(slines, zlines, 
            tag_mode=tag_mode, 
            beam=4, lenpen=2.0, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=3, guided=True)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
