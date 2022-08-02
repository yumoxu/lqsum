import argparse
import codecs
import json
import numpy as np
import re
# Get a counter for the iterations
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import Counter
import string
import io

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import *
stemmer = PorterStemmer()

from nltk.tokenize import sent_tokenize

print("Loaded libraries...")

parser = argparse.ArgumentParser(
    description="Builds an extractive summary from a json prediction.")

parser.add_argument('-src', required=True, type=str,
                    help="""Path of the src file""")
parser.add_argument('-tgt', required=True, type=str,
                    help="""Path of the tgt file""")

parser.add_argument('-output', type=str,
					default='data/processed/multicopy',
                    help="""Path of the output files""")

parser.add_argument('-prune', type=int, default=200,
                   help="Prune to that number of words.")
parser.add_argument('-num_examples', type=int, default=100000,
                   help="Prune to that number of examples.")
parser.add_argument('--full', action='store_true', default=False,
                   help='only the earliest appearance of the same phrase will be annotated.')
parser.add_argument('--stem', action='store_true', default=False,
                   help='Find LCS after stemming.')
parser.add_argument('--compare_word_seq', action='store_true', default=False,
                   help='Do matching between source & target word sequences.')
parser.add_argument('--tag_stop_words_as_zero', action='store_true', default=False,
                   help='Tag LCS as zero, if the LCS found has a length of 1 and is in the stop word list.')
parser.add_argument('--first_tgt_line', action='store_true', default=False,
                   help='Use only the first summary line as target.')


opt = parser.parse_args()

def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end+1])


def compile_sublist(start, end, split):
    if start == end:
        return [split[start]]
    return split[start:end+1]


def format_json(s):
    return json.dumps({'sentence':s})+"\n"

def splits(s, num=200):
    return s.split()[:num]


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


def fix_punc(words):
	punc_fixed = []
	for word in words:
		while True:  # recurrently remove the punctuation from end
			if len(word) == 1 or word[-1] not in string.punctuation:
				break
			word = word[:-1]

		while True:  # recurrently remove the punctuation from start
			if len(word) == 1 or word[0] not in string.punctuation:
				break
			word = word[1:]

		punc_fixed.append(word)
	assert len(punc_fixed) == len(words)
	return punc_fixed


def is_sublist(a, b):
	"""
		Check if a is a sublist of b.

	"""
	assert isinstance(a, list) and isinstance(b, list)
	if len(a) > len(b):
		return False
	for i in range(0, len(b) - len(a) + 1):
		if b[i:i+len(a)] == a:
			return True
	return False


def is_functional_word(word):
	if word in stop_words:
		return True
	return False


def is_punc(word):
	"""
		Judge whether word is composed of punctuations.
	"""
	for token in word:
		if token not in string.punctuation:
			return False
	return True


def make_BIO_tgt_with_stem(s, t, tag_stop_words_as_zero):
    """
		Do matching between source word sequence and target token sequence.

        s: a list of words
        t: a summary string

		One bad case example:
			- src: 8 is a word
			- tgt: 68 is a word
			- 8 will be annotated as positive as a subsequence of 68
    """
    punc_fixed_s = fix_punc(s)
    punc_fixed_t = fix_punc(t.split())

    ssplit = [stemmer.stem(word) for word in punc_fixed_s]
    tsplit = [stemmer.stem(word) for word in punc_fixed_t]
    t = ' '.join(tsplit)
    
    for word, stemmed in zip(s, ssplit):
	    if not stemmed:
	        print(f'word: {word} -> {stemmed}')
    
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        if searchstring in t and endix < len(ssplit)-1:  # match but still extend to check if it is the longest
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
                    tag = "1"
                    if tag_stop_words_as_zero and startix == endix-1 and punc_fixed_s[startix] in stop_words:  # one word and is stop word
                        tag = "0"
                        # print(f'Tag stop words with 0: {s[startix]}')
                    matches.extend([tag]*(endix-startix))
                    matchstrings[full_string] +=1
                #endix += 1
            startix = endix
    return " ".join(matches)


def make_BIO_tgt_with_stem_word_seq(s, t, tag_stop_words_as_zero):
    """
		Do matching between source & target word sequences.

        s: a list of words
        t: a summary string

		Do matching between source and target word lists.
    """
    punc_fixed_s = fix_punc(s)
    punc_fixed_t = fix_punc(t.split())

    ssplit = [stemmer.stem(word) for word in punc_fixed_s]
    tsplit = [stemmer.stem(word) for word in punc_fixed_t]
    # t = ' '.join(tsplit)
    
    for word, stemmed in zip(s, ssplit):
	    if not stemmed:
	        print(f'word: {word} -> {stemmed}')
    
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchsublist = compile_sublist(startix, endix, ssplit)
        if is_sublist(searchsublist, tsplit) and endix < len(ssplit)-1:  # match but still extend to check if it is the longest
            endix +=1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:#-1:
                matches.extend(["0"] * (endix-startix+1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix-1, ssplit)
                if matchstrings[full_string] >= 1:
                    matches.extend(["0"]*(endix-startix))
                else:
                    tag = "1"
                    if startix == endix-1:
                        _word = punc_fixed_s[startix]
                        if tag_stop_words_as_zero and is_functional_word(_word):  # one word and is stop word
                            tag = "0"
                        if is_punc(_word):
                            tag = '0'
					
                    matches.extend([tag]*(endix-startix))
                    matchstrings[full_string] += 1
                #endix += 1
            startix = endix
    return " ".join(matches)


def main_original_with_buffer():
	"""
		This function is revised to not skip records that are too short.
        Instead, tags of all 0 are produced.
	"""
	lcounter = 0
	max_total = opt.num_examples

	SOURCE_PATH = opt.src
	TARGET_PATH = opt.tgt

	NEW_TARGET_PATH = opt.output + ".txt"
	PRED_SRC_PATH = opt.output + ".pred.txt"
	PRED_TGT_PATH = opt.output + ".src.txt"

	with codecs.open(SOURCE_PATH, 'r', "utf-8") as sfile:
	    for ix, l in enumerate(sfile):
	        lcounter +=1
	        if lcounter >= max_total:
	            break

	sfile = codecs.open(SOURCE_PATH, 'r', "utf-8")
	tfile = codecs.open(TARGET_PATH, 'r', "utf-8")
	outf = codecs.open(NEW_TARGET_PATH, 'w', "utf-8", buffering=1)
	outf_tgt_src = codecs.open(PRED_SRC_PATH, 'w', "utf-8", buffering=1)
	outf_tgt_tgt = codecs.open(PRED_TGT_PATH, 'w', "utf-8", buffering=1)

	actual_lines = 0
	for ix, (s, t) in tqdm(enumerate(zip(sfile,tfile)), total=lcounter):
	    ssplit = splits(s, num=opt.prune)
	    # Skip empty lines
	    if len(ssplit) < 2 or len(t.split()) < 2:
	        tgt = " ".join(["0"] * len(ssplit))
			# continue
	    else:
	        # actual_lines += 1
	        tgt = make_BIO_tgt(ssplit,t)
	    
	    # Format for allennlp
	    for token, tag in zip(ssplit, tgt.split()):
	        outf.write(token+"###"+tag + " ")
	    outf.write("\n")
	    # Format for predicting with allennlp
	    outf_tgt_src.write(format_json(" ".join(ssplit)))
	    outf_tgt_tgt.write(tgt + "\n")
	    if actual_lines >= max_total:
	        break

	sfile.close()
	tfile.close()
	outf.close()
	outf_tgt_src.close()
	outf_tgt_tgt.close()


def main_debug():
	"""
		This is for debugging.
		
		We keep the original codecs way to open files, and iterate two files.txt
		However, the output is still in wrong order, and tags are not accurate 
		i.e, lots of all-zero cases after line 20,0000. 

	"""
	
	lcounter = 0
	with codecs.open(opt.src, 'r', "utf-8") as sfile:
	    for ix, l in enumerate(sfile):
	        lcounter +=1

	sfile = codecs.open(opt.src, 'r', "utf-8")
	tfile = codecs.open(opt.tgt, 'r', "utf-8")

	import io
	with io.open(opt.output + ".txt", mode='a', encoding="utf-8") as outf, \
		io.open(opt.output + ".pred.txt", mode='a', encoding="utf-8") as outf_tgt_src, \
		io.open(opt.output + ".src.txt", mode='a', encoding="utf-8") as outf_tgt_tgt:
		
		# for ix, (s, t) in tqdm(enumerate(zip(source_lines, target_lines)), total=n_lines):
		for ix, (s, t) in tqdm(enumerate(zip(sfile,tfile)), total=lcounter):
			ssplit = splits(s, num=opt.prune)
			if len(ssplit) < 2 or len(t.split()) < 2:
				tgt = " ".join(["0"] * len(ssplit))
			else:
				tgt = make_BIO_tgt(ssplit,t)
			
			# Format for allennlp
			for token, tag in zip(ssplit, tgt.split()):
				outf.write(token+"###"+tag + " ")
			outf.write("\n")

			# Format for predicting with allennlp
			outf_tgt_src.write(format_json(" ".join(ssplit)))
			outf_tgt_tgt.write(tgt + "\n")

	sfile.close()
	tfile.close()


def get_first_line(line):
	return sent_tokenize(line)[0]


def main():
	"""
		This function is revised to not skip records that are too short.
        Instead, tags of all 0 are produced.

		No buffer is used to ensure correct sample order.
	"""
	source_lines = io.open(opt.src, encoding="utf-8").readlines()
	target_lines = io.open(opt.tgt, encoding="utf-8").readlines()
	n_lines = len(source_lines)

	# with io.open(opt.output + ".txt", mode='a', encoding="utf-8") as outf, \
	# 	io.open(opt.output + ".pred.txt", mode='a', encoding="utf-8") as outf_tgt_src, \
	# 	io.open(opt.output + ".src.txt", mode='a', encoding="utf-8") as outf_tgt_tgt:
	with io.open(opt.output + ".txt", mode='a', encoding="utf-8") as outf, \
		io.open(opt.output + ".json", mode='a', encoding="utf-8") as outf_json, \
        io.open(opt.output + ".src", mode='a', encoding="utf-8") as outf_src, \
		io.open(opt.output + ".tgt", mode='a', encoding="utf-8") as outf_tgt:
		for ix, (s, t) in tqdm(enumerate(zip(source_lines, target_lines)), total=n_lines):
			s = s.strip('\n')
			t = t.strip('\n')

			ssplit = splits(s, num=opt.prune)
			if opt.first_tgt_line:
				t = get_first_line(t)
			
			if len(ssplit) < 2 or len(t.split()) < 2:
				tgt = " ".join(["0"] * len(ssplit))
			else:
				if opt.full:
					tgt = make_BIO_tgt_with_all(ssplit, t)
				elif opt.stem:
					if opt.compare_word_seq:
						tgt = make_BIO_tgt_with_stem_word_seq(ssplit, t, tag_stop_words_as_zero=opt.tag_stop_words_as_zero)
					else:
						tgt = make_BIO_tgt_with_stem(ssplit, t, tag_stop_words_as_zero=opt.tag_stop_words_as_zero)
				else:
					tgt = make_BIO_tgt(ssplit, t)
			
			# Format for allennlp
			for token, tag in zip(ssplit, tgt.split()):
				outf.write(token+"###"+tag + " ")
			outf.write("\n")

			# Format for predicting with allennlp
			# outf_tgt_src.write(format_json(" ".join(ssplit)))
			# outf_tgt_tgt.write(tgt + "\n")

			# Format for predicting with allennlp
			outf_json.write(format_json(" ".join(ssplit)))
			outf_src.write(s + "\n")
			outf_tgt.write(tgt + "\n")


if __name__ == "__main__":
    main()
	# print(fix_punc(["'French", "'", "natural,", 'P."', "''''x.!"]))
