import subprocess
import nltk
import numpy as np
import argparse
import sys
import functools
print = functools.partial(print, flush=True)

JAVA_CLASS_PATH = '/om/user/pqian/tools/stanford-tregex-2018-10-16/*'


def extract_output(line):
    """
    Extract the index of the tree and the matched tree fragment from a line of Tregex outout.
    """
    line = line.strip()
    index = None
    fragment = None
    for k, c in enumerate(line):
        if c == ':':
            index = line[:k]
            fragment = line[(k+1):].strip()
            break
    assert index is not None
    return int(index), fragment


def tregex_match(pattern, trees, from_file=False):
    if from_file:
        tregex_command = ['java', '-cp', JAVA_CLASS_PATH, 'edu.stanford.nlp.trees.tregex.TregexPattern', 
                            '-filter', '-o', '-t', '-s', '-n', pattern, trees]
        output = subprocess.run(tregex_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='utf-8')
    else:
        tregex_command = ['java', '-cp', JAVA_CLASS_PATH, 'edu.stanford.nlp.trees.tregex.TregexPattern', 
                            '-filter', '-o', '-t', '-s', '-n', pattern]
        output = subprocess.run(tregex_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, input='\n'.join(trees), encoding='utf-8')

    matched_lines = output.stdout.splitlines()

    matched_trees = {}
    for line in matched_lines:
        tree_id, fragment = extract_output(line)
        if tree_id not in matched_trees:
            matched_trees[tree_id] = []
        matched_trees[tree_id].append(fragment)

    matched_tree_ids = sorted(list(matched_trees.keys()))

    return matched_tree_ids, matched_trees


def load_data(path, nStimuli, nSample):
    with open(path) as f:
        lines = f.readlines()

    data = {}

    for i in range(nStimuli):
        stimulus = lines[(nSample+2)*i].strip()
        samples = [line.strip() for line in lines[((nSample+2)*i+1):((nSample+2)*i+nSample+1)]]
        data[stimulus] = samples
    return data


def load_stimuli(path):
    with open(path) as f:
        stimuli = f.readlines()
    stimuli = [stimulus.strip() for stimulus in stimuli]
    stimuli = [stimulus for stimulus in stimuli if stimulus != '']
    return stimuli


def get_pattern(test_name, stimulus):
    fragments = stimulus.split('%')
    if test_name in set(['clause_VP', 'clause_VP-with-PP-adjunct', 'clause_VP-with-adjunct-long', 
                        'clause_VP-with-complement', 'clause_VP-with-complement-long']):
        tokens = fragments[2].strip().split()
        target_words = {'v1':tokens[0], 'v2':tokens[-1]}
        pattern = "VP (<< {v2}) & [$, (NP << (VP << {v1})) | ,, (@S|SBAR < (VP << {v1}))]".format(**target_words)
    elif test_name == 'clause_VP-gerund':
        tokens = fragments[2].strip().split()
        target_words = {'verb':tokens[0], 'aux':tokens[-1]}
        pattern = "SBAR [< (S <+(VP) (VP << {verb} < (SBAR < (S < (VP <<, {aux}))))) | (< (S < (VP << {verb}) ) >> (@NP|S $. (VP <<, {aux}))) | (< (S <+(VP) (VP << {verb})) >+(NP) (S < (VP <<, {aux})))]".format(**target_words)
    elif test_name == 'clause_phrasal-verb-with-subj':
        tokens = nltk.word_tokenize(fragments[2])
        target_words = {'verb':tokens[-3], 'particle':tokens[-2]} # the last token is punctuation
        pattern = "SBAR [ (<, @WHNP|WHADVP <- (S << (VP << {verb} <<- {particle}))) | ($, NP > NP < (S << (VP << {verb} <<- {particle})))]".format(**target_words)
    elif test_name == 'clause_phrasal-verb':
        tokens = nltk.word_tokenize(fragments[2])
        target_words = {'verb':tokens[-3], 'particle':tokens[-2]} # the last token is punctuation
        pattern = "S [(<< SBAR [ (<, @WHNP|WHADVP <- (S << (VP << {verb} <<- {particle}))) | ($, NP > NP < (S << (VP << {verb} <<- {particle})))]) | << (VP [ (<<, @was|were|is|am|are|be|been|being << {verb} <<- {particle}) | (< (ADJP << {verb} <<- {particle})) ]) ]".format(**target_words)
    elif test_name == 'clause_resultative':
        tokens = nltk.word_tokenize(fragments[-1])
        target_words = {'w':tokens[0], 'adj':tokens[-2]} # the last token is punctuation
        pattern = "S <, (NP << {w} ) <- (ADJP << {adj})".format(**target_words)
    elif test_name == 'clause_resultative-long':
        tokens = nltk.word_tokenize(fragments[-1])
        target_words = {'w1':tokens[0], 'w2':tokens[1], 'w3':tokens[2], 'adj':tokens[-2]} # the last token is punctuation
        pattern = "S [ (<, (NP << {w1} << {w2} << {w3}) <- (ADJP << {adj})) | (< (NP << EX $. (VP <- (NP <, (NP << {w1} << {w2} << {w3}) <- (ADJP << {adj}))))) ]".format(**target_words)
    elif test_name == 'agreement_emb-subj-long':
        tokens = fragments[2].strip().split()
        target_words = {'nsubj':tokens[-1], 'predicate':fragments[4].strip()}
        pattern = "S <, (NP << {nsubj}) < VP !<<, {predicate}".format(**target_words)
    elif test_name == 'agreement_subj' or test_name == 'agreement_subj-long' or test_name == 'agreement_subj-with-PP':
        tokens = fragments[0].strip().split()
        target_words = {'det':tokens[0], 'nsubj':tokens[1], 'predicate':fragments[2].strip()}
        singular_predicates = set(['has', 'is', 'was'])
        plural_predicates = set(['have', 'are', 'were'])
        if target_words['predicate'] in singular_predicates:
            pattern = "NP [ (<<, {det} << {nsubj} $. (VP !<<, {predicate})) | (<<, {det} < (@/NN.?/ < {nsubj} $.. @NN ) $.. (VP <<, {predicate})) ]".format(**target_words)
        elif target_words['predicate'] in plural_predicates:
            pattern = "NP [ (<<, {det} << {nsubj} $. (VP !<<, {predicate})) | ((<<, {det} << {nsubj} << CC) $.. (VP <<, {predicate})) ]".format(**target_words)
        else:
            raise NotImplementedError
    elif test_name == 'agreement_subj-with-coord':
        tokens = fragments[0].strip().split()
        target_words = {'det':tokens[0], 'nsubj':tokens[1], 'conj':tokens[-1], 'predicate':fragments[2].strip()} 
        pattern = "S < (NP <<, {det} << {nsubj} << {conj}) < (VP !<<, {predicate})".format(**target_words)
    elif test_name == 'coord_S':
        tokens = fragments[2].strip().split()
        target_words = {'n1':tokens[0], 'conj':tokens[1], 'det':tokens[2], 'n2':tokens[3]}
        pattern = "S < (CC < {conj} $, (S < (VP <<- {n1})) $. (S <<, (NP << {det} << {n2}) < VP ))".format(**target_words)
    elif test_name == 'coord_VP':
        tokens = fragments[2].strip().split()
        target_words = {'noun':tokens[0], 'conj':tokens[1], 'verb':tokens[2]}
        pattern = "VP < (CC < {conj} $, (VP <<- {noun}) $. (VP <<, {verb}))".format(**target_words)
    elif test_name == 'coord_either':        
        tokens = fragments[2].strip().split()
        target_words = {'verb':tokens[1]}
        pattern = "VP << either << (CC < or $,, (VP << {verb}) $.. VP)".format(**target_words)
    elif test_name == 'coord_neither':        
        tokens = fragments[2].strip().split()
        target_words = {'verb':tokens[1]}
        pattern = "S [ < (VP << neither << (CC < nor $,, (VP << {verb}) $.. VP)) | (< (CC < neither) $.. (CC < nor) $. @S|SINV) | (<< neither << @nor|or )]".format(**target_words)
    elif test_name == 'coord_emb-NP':
        tokens = nltk.word_tokenize(fragments[2])
        target_words = {'w1':tokens[0], 'verb':tokens[-3], 'particle':tokens[-2]} # the last token is punctuation
        pattern = "SBAR [ (<, @WHNP|WHADVP <- (S << (NP < (CC < @and|or $, (NP << {w1}) $. NP)) << (VP << {verb} <<- {particle}))) | ($, NP > NP < (S << (NP < (CC < @and|or $, (NP << {w1}) $. NP)) << (VP << {verb} <<- {particle})))]".format(**target_words)
    elif test_name == 'coord_emb-VP':
        tokens = nltk.word_tokenize(fragments[2])
        target_words = {'coordverb':tokens[0], 'verb':tokens[-1]}
        pattern = "VP < (CC < and $, (VP << {coordverb}) $. VP) .. (VP << {verb})".format(**target_words)
    elif test_name == 'coord_gap-NP':
        tokens = fragments[2].strip().split()
        target_words = {'det':tokens[1], 'n1':tokens[2], 'n2':tokens[-1]}
        pattern = "SBAR << (WHNP << what) <- (S < (NP < (CC < and $, (NP << {det} << {n1}) $. (NP << {n2}))) < (VP !< NP))".format(**target_words)
    elif test_name == 'gap_adjunct':
        tokens = fragments[2].strip().split() + nltk.word_tokenize(fragments[4])
        target_words = {'wh':tokens[0], 'det':tokens[1], 'w1':tokens[2], 'w2':tokens[3]}
        pattern = "SBAR < (WHADVP << {wh}) < (S <, (NP <<, {det} !< (SBAR << {w1} << {w2}) !$. (VP <<, {w1} << {w2})))".format(**target_words)
    elif test_name == 'gap_obj':
        tokens = fragments[2].strip().split() + nltk.word_tokenize(fragments[4])
        target_words = {'filler':tokens[0], 'det':tokens[1], 'noun':tokens[-2]}
        pattern = "SBAR < (WHNP << {filler}) <- (S < (NP <<, {det}) < (VP !< (NP <<- {noun})))".format(**target_words)
    elif test_name == 'gap_subj':
        tokens = fragments[2].strip().split() + nltk.word_tokenize(fragments[4])
        target_words = {'filler':tokens[0], 'w1':tokens[1], 'w2':tokens[2]}
        pattern = "SBAR < (WHNP << {filler}) !< (S << (VP << {w1} << {w2} < NP $, NP) )".format(**target_words)
    elif test_name == 'gap_phrasal-verb':
        tokens = fragments[2].strip().split() + nltk.word_tokenize(fragments[4])
        target_words = {'filler':tokens[0], 'verb':tokens[1], 'particle':tokens[2]}
        pattern = "SBAR [ (!<<,{filler} < (S <+(VP) (VP << {verb} << {particle}))) | (<<,{filler} < (S <+(VP) (VP <<, @was|were|been|be|am|is|are << {verb} << {particle}))) ]".format(**target_words)
    else:
        raise NotImplementedError

    return pattern


def eval_syntactic_structure(stimuli_path, data_path, test_name, verbose=False):
    stimuli = load_stimuli(stimuli_path)
    nStimuli = len(stimuli)
    nSample = 35
    data = load_data(data_path, nStimuli, nSample)

    matched_count = 0
    total_count = 0

    matched_rs = []

    for stimulus in stimuli:
        trees = data[stimulus]
        pattern = get_pattern(test_name, stimulus)
        matched_tree_ids, _ = tregex_match(pattern, trees=trees, from_file=False)
        if verbose:
            matched_tree_id_set = set(matched_tree_ids)
            for j in range(len(trees)):
                matched_flag = 1 if j+1 in matched_tree_id_set else 0
                print("{}\t{}\t{}\t{}".format(test_name, stimulus, trees[j], matched_flag), file=sys.stderr)
        matched_count += len(matched_tree_ids)
        total_count += len(trees)
        matched_rs.append([stimulus, len(matched_tree_ids), len(trees)])
    return matched_rs    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fpath', type=str, default=None,
                        help='Path to parsed output sentences.')
    parser.add_argument('-t','--test_name', type=str, default=None,
                        help='Name of the test stimuli.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print out detailed evaluation results.')
    args = parser.parse_args()

    test_name = args.test_name
    stimuli_path = 'stimuli/exp1/{}.txt'.format(test_name)
    data_path = args.fpath
    
    eval_rs = eval_syntactic_structure(stimuli_path, data_path, test_name, verbose=args.verbose)
    acc = np.sum([rs[1] for rs in eval_rs])/np.sum([rs[2] for rs in eval_rs])

    for i in range(len(eval_rs)):
        rs = eval_rs[i]
        item_acc = rs[1]/rs[2]
        print('{:<32} {} {} {}/{}'.format(test_name, i, item_acc, rs[1], rs[2]))

