import torch
from fairseq.models.bart import BARTModel

import sys
from tqdm import tqdm

bart = BARTModel.from_pretrained(
    sys.argv[3],
    checkpoint_file=sys.argv[4],
    data_name_or_path=sys.argv[5]
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 16

if 'cnndm' in sys.argv[2]:
    testset = 'cnndm'
    test_size = 11490
    min_len = 55
    max_len_b = 140
    beam = 4
    lenpen = 2.0
elif 'wikicat' in sys.argv[2]:
    testset = 'wikicat'
    test_size = 3000
    min_len = 55
    max_len_b = 140
    beam = 5
    lenpen = 2.0
elif 'xsum' in sys.argv[2]:
    testset = 'xsum'
    test_size = 11334
    min_len = 10
    max_len_b = 60
    beam = 6
    lenpen = 1.0
elif 'wikiref' in sys.argv[2]:
    testset = 'wikiref'
    test_size = 12000
    min_len = 35
    max_len_b = 90
    beam = 4
    lenpen = 2.0
elif 'debatepedia' in sys.argv[2]:
    testset = 'debatepedia'
    test_size = 1000
    min_len = 5
    max_len_b = 25
    beam = 6
    lenpen = 1.0
elif 'duc' in sys.argv[2]:
    testset = 'duc'
    use_marge = False
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
            if year in sys.argv[2]:
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
elif 'tdqfs' in sys.argv[2]:
    testset = 'tdqfs'
    use_marge = 'marge' in sys.argv[2] 
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
    raise ValueError(f'Test set unrecognized from the path: {sys.argv[2]}')

print(f'testset: {testset}, min_len: {min_len}, min_len_b: {max_len_b}, beam: {beam}, lenpen: {lenpen}')


with open(sys.argv[1]) as source, open(sys.argv[2], 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]

    # for sline in source:
    for sline in tqdm(source, total=test_size):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
