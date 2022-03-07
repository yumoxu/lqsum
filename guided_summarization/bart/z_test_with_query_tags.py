import torch
from fairseq.models.bart import (
    GuidedBARTModelwithTagging, 
    GuidedBARTModelwithTaggingAndStdDec, 
    GuidedBARTModelwithTaggingAndStdEncDec,
)

import sys
from tqdm import tqdm
import logging

from icecream import install
install()

model_class = GuidedBARTModelwithTagging  # GuidedBARTModelwithTagging, GuidedBARTModelwithTaggingAndStdDec, GuidedBARTModelwithTaggingAndStdEncDec
proc_no_match = False

if model_class is GuidedBARTModelwithTaggingAndStdDec:
    stdenc, stddec = False, True
elif model_class is GuidedBARTModelwithTaggingAndStdEncDec:
    stdenc, stddec = True, True
else:
    stdenc, stddec = False, False

bart = model_class.from_pretrained(
    sys.argv[4],
    checkpoint_file=sys.argv[5],
    data_name_or_path=sys.argv[6]
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 8

tag_mode = sys.argv[7]  # e.g., 'query_1.0'
assert 'query' in tag_mode, f'Invalid tag_mode: {tag_mode}'

if 'cnndm' in sys.argv[3]:
    testset = 'cnndm'
    test_size = 11490
    min_len = 55
    max_len_b = 140
    beam = 4
    lenpen = 2.0
elif 'wikicat' in sys.argv[3]:
    testset = 'wikicat'
    test_size = 3000
    min_len = 55
    max_len_b = 140
    beam = 5
    lenpen = 2.0
elif 'xsum' in sys.argv[3]:
    testset = 'xsum'
    test_size = 11334
    min_len = 10
    max_len_b = 60
    beam = 6
    lenpen = 1.0
elif 'wikiref' in sys.argv[3]:
    testset = 'wikiref'
    test_size = 12000
    min_len = 35
    max_len_b = 90
    beam = 4
    lenpen = 2.0
elif 'debatepedia' in sys.argv[3]:
    testset = 'debatepedia'
    test_size = 1000
    min_len = 5
    max_len_b = 25
    beam = 6
    lenpen = 1.0
elif 'duc' in sys.argv[3]:
    testset = 'duc'
    use_marge = 'marge' in sys.argv[3] 
    year2docs = {
        '2005': 1593,
        '2006': 1250,
        '2007': 1125,
    }
    if use_marge:
        test_size = 50
        min_len = 300
        max_len_b = 400
        beam = 5
        lenpen = 0.9
    else:
        test_size = 1000
        for year in year2docs.keys():
            if year in sys.argv[3]:
                test_size = year2docs[year]
                break
    
        # min_len = 55
        # max_len_b = 140
        # beam = 4
        # lenpen = 2.0
        
        min_len = 10
        max_len_b = 60
        beam = 6
        lenpen = 1.0
elif 'tdqfs' in sys.argv[3]:
    testset = 'tdqfs'
    use_marge = 'marge' in sys.argv[3] 
    if use_marge:
        test_size = 50
        min_len = 300
        max_len_b = 400
        beam = 5
        lenpen = 0.9
    else:
        test_size = 7099        
        min_len = 10
        max_len_b = 60
        beam = 6
        lenpen = 1.0
else:
    raise ValueError(f'Test set unrecognized from the path: {sys.argv[3]}')

print(f'testset: {testset}, min_len: {min_len}, min_len_b: {max_len_b}, beam: {beam}, lenpen: {lenpen}')

with open(sys.argv[1]) as source, open(sys.argv[2]) as zs, open(sys.argv[3], 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    zline = zs.readline().strip()
    zlines = [zline]

    for sline, zline in tqdm(zip(source, zs), total=test_size):
        # if count < 840:
        #     slines = [sline]
        #     zlines = [zline]
        #     count += 1
        #     continue
        
        if count % bsz == 0:
            # ic(count)
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, zlines, 
                    tag_mode=tag_mode,
                    beam=beam, 
                    use_bped_input=True,
                    lenpen=lenpen, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=3, 
                    guided=True, stddec=stddec, stdenc=stdenc, 
                    proc_no_match=proc_no_match)

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
            beam=beam, 
            use_bped_input=True,
            lenpen=lenpen, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=3, 
            guided=True, stddec=stddec, stdenc=stdenc,
            proc_no_match=proc_no_match)
        
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
