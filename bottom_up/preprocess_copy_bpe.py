import argparse
import json
import numpy as np
import re
# Get a counter for the iterations
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import Counter
import string

"""
	This file takes two BPE-ed files as inputs, and calculates their overlaps.

    Usually one file is a doc file (x), 
    and the other file can either be a summary file (y), or a query file (q). 

"""

print("Loaded libraries...")

parser = argparse.ArgumentParser(
    description="Builds an extractive summary from a json prediction.")

parser.add_argument('-src', required=True, type=str,
                    help="""Path of the src file""")
parser.add_argument('-tgt', required=True, type=str,
                    help="""Path of the tgt file""")
parser.add_argument('-output', required=True, type=str,
                    help="""Path of the output files""")
parser.add_argument('--full', action='store_true', default=False,
                   help='only the earliest appearance of the same phrase will be annotated.')
parser.add_argument('--proc_no_match', action='store_true', default=False,
                   help='for query as target and no match is found in source')

parser.add_argument('--no_match_only', action='store_true', default=False,
                   help='get a subset of records w/o query match in the source.')
parser.add_argument('--match_only', action='store_true', default=False,
                   help='get a subset of records with at least one query match in the source.')

opt = parser.parse_args()

def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end+1])


def format_json(s):
    return json.dumps({'sentence':s})+"\n"


def make_BIO_tgt(s, t):
    # tsplit = t.split()
    ssplit = s#.split()
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        if searchstring in t \
            and endix < len(ssplit)-1:
            endix +=1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:#-1:
                matches.extend(["0"] * (endix-startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix-1, ssplit)
                if matchstrings[full_string] >= 1:
                    matches.extend(["0"]*(endix-startix))
                else:
                    matches.extend(["1"]*(endix-startix))
                    matchstrings[full_string] +=1
                #endix += 1
            startix = endix
    return " ".join(matches)


def make_BIO_tgt_with_all(s, t):
    # tsplit = t.split()
    ssplit = s#.split()
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        if searchstring in t \
            and endix < len(ssplit)-1:
            endix +=1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:#-1:
                matches.extend(["0"] * (endix-startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix-1, ssplit)
                matches.extend(["1"]*(endix-startix))
                matchstrings[full_string] +=1
                #endix += 1
            startix = endix
    return " ".join(matches)


def main():
	"""
		This function is revised to not skip records that are too short.
        Instead, tags of all 0 are produced.

		No buffer is used to ensure correct sample order.
	"""
	import io
	source_lines = io.open(opt.src, encoding="utf-8").readlines()
	target_lines = io.open(opt.tgt, encoding="utf-8").readlines()
	n_lines = len(source_lines)

	save_indices = []

	with io.open(opt.output + ".txt", mode='a', encoding="utf-8") as outf, \
		io.open(opt.output + ".json", mode='a', encoding="utf-8") as outf_json, \
        io.open(opt.output + ".src", mode='a', encoding="utf-8") as outf_src, \
		io.open(opt.output + ".tgt", mode='a', encoding="utf-8") as outf_tgt, \
        io.open(opt.output + ".index", mode='a', encoding="utf-8") as outf_idx:
		
		for ix, (s, t) in tqdm(enumerate(zip(source_lines, target_lines)), total=n_lines):
			s = s.strip('\n')
			t = t.strip('\n')
			ssplit = s.split()
			if len(ssplit) < 2 or len(t.split()) < 2:
				tgt = " ".join(["0"] * len(ssplit))
			else:
				if opt.full:
					tgt = make_BIO_tgt_with_all(ssplit,t)
				else:
					tgt = make_BIO_tgt(ssplit,t)
			
			no_match = sum(int(tag) for tag in tgt.split()) == 0
			if (opt.no_match_only and not no_match) or (opt.match_only and no_match):
				continue
            
			save_indices.append(str(ix))
			if opt.proc_no_match and no_match:
				ssplit = t.split() + ssplit
				s = " ".join(ssplit)
				new_tags = ["1"] * len(t.split()) + tgt.split()
				tgt = " ".join(new_tags)
				assert len(ssplit) == len(new_tags)
				print(ix, len(ssplit), s)
				print(ix, len(tgt.split()), tgt)

			# Format for allennlp
			assert len(ssplit) == len(tgt.split())
			for token, tag in zip(ssplit, tgt.split()):
				outf.write(token+"###"+tag+" ")
			outf.write("\n")

            # Format for predicting with allennlp
			outf_json.write(format_json(" ".join(ssplit)))
			outf_src.write(s + "\n")
			outf_tgt.write(tgt + "\n")
        
		outf_idx.write('\n'.join(save_indices))


if __name__ == "__main__":
    main()
