import torch
from fairseq.models.bart import BARTModel

import sys

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

MIN_LEN = 55  # default: 55

with open(sys.argv[1]) as source, open(sys.argv[2], 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=MIN_LEN, no_repeat_ngram_size=3,
                    use_bped_input=True)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=MIN_LEN, no_repeat_ngram_size=3,
            use_bped_input=True)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
