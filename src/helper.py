import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer

PTB_detokenizer = TreebankWordDetokenizer()


CONTRACTIONS= ["'s", "'ve", "'d", "'ll", "'t", "n't", "'m", "'re"]
CONTRACTIONS_SET = set(CONTRACTIONS + [tok.upper() for tok in CONTRACTIONS])

NO_PREFIX_SPACE_TOKEN_SET = set(CONTRACTIONS + [tok.upper() for tok in CONTRACTIONS] + [',', '.', ';', ':', '!', '?', ')'])

def get_detokenized_str(toks):
    detokenized_str = PTB_detokenizer.detokenize(toks, convert_parentheses=True)
    if toks[-1] in CONTRACTIONS_SET:
        detokenized_str = detokenized_str.replace(' '+toks[-1], toks[-1])
    return detokenized_str


def add_gaps(tokens, BLANK=None, FILLER=None, special_format=None, min_nSpan=2, max_nSpan=9, rng=None):
    """
    Randomly replace parts of the sentence with BLANK symbol, and append the original
    contents of the blanks in order separated by FILLER symbol.
    The number of blanks may vary between min_nBlank and max_nBlank.
    Assume the input sentence is pretokenized and separated by whitespace.

    The special_format argument specifies ways of constructing the context and answer pairs:
    For T5, the symbols for the blanks are indexed, e.g. <extra_id_0>, <extra_id_1>, etc;
    In general, for ILM and BART, BLANK as the symbol for all the blanks, and FILLER as the delimiter of filler spans.
    """
    nToken = len(tokens)
    max_nSpan = np.min([max_nSpan, nToken])

    if rng is None:
        nSpan = np.random.choice(np.arange(min_nSpan, max_nSpan+1))
        span_end_indice = sorted(np.random.choice(np.arange(nToken-1), nSpan-1, replace=False)) + [nToken-1]
        is_blank_first = np.random.randint(2)
    else:
        nSpan = rng.choice(np.arange(min_nSpan, max_nSpan+1))
        span_end_indice = sorted(rng.choice(np.arange(nToken-1), nSpan-1, replace=False)) + [nToken-1]
        is_blank_first = rng.randint(2)        

    context = []
    answer = []

    if special_format is None:
        for i in range(nSpan):
            start_index = span_end_indice[i-1] + 1 if i > 0 else 0
            end_index = span_end_indice[i] + 1
            if (i + is_blank_first) % 2 == 1:
                answer += get_detokenized_str(tokens[start_index:end_index]).split() + [FILLER]
                context += [BLANK]
            else:
                context += get_detokenized_str(tokens[start_index:end_index]).split()
    elif special_format == 'T5':
        blank_index = 0
        for i in range(nSpan):
            start_index = span_end_indice[i-1] + 1 if i > 0 else 0
            end_index = span_end_indice[i] + 1
            if (i + is_blank_first) % 2 == 1:
                answer += ['<extra_id_{}>'.format(blank_index)] + get_detokenized_str(tokens[start_index:end_index]).split()
                context += ['<extra_id_{}>'.format(blank_index)]
                blank_index += 1
            else:
                context += get_detokenized_str(tokens[start_index:end_index]).split()
    else:
        raise NotImplementedError
    return context, answer


def add_indexed_masks(context):
    text_template = []
    fragments = context.strip().split('%')
    mask_index = 0
    for i, fragment in enumerate(fragments):
        if i % 2 == 0:
            text_template.append(fragment)
        else:
            text_template.append('<extra_id_{}>'.format(mask_index))
            mask_index += 1
    return ''.join(text_template)


def test_add_gaps(path, BLANK, FILLER, special_format=None):
    with open(path) as f:
        for line in f:
            line = line.strip()
            context, answer = add_gaps(line.split(), BLANK=BLANK, FILLER=FILLER, special_format=special_format)
            stimulus = ' '.join(context + ['[SEP]'] + answer)
            print(line)
            print(stimulus)


def get_batches(lines, batch_size):
    if len(lines) % batch_size == 0:
        num_batches = len(lines) // batch_size
    else:
        num_batches = len(lines) // batch_size + 1
    batches = []
    for i in range(num_batches):
        start_index = i*batch_size
        end_index = (i+1)*batch_size
        batch = lines[start_index:end_index]
        batches.append(batch)
    return batches


def load_data(path):
    """
    Load the text corpus into a list of lines. Each line is a list of tokens.
    """
    lines = open(path).readlines()
    lines = [line.strip().split() for line in lines]  
    return lines


def filter_data(lines):
    """
    Filter out lines that are too short or too long.
    """
    return [line for line in lines if len(line) >= 8 and len(line) <= 60]

def sample_from_scores(logits, top_k=50):
    kth_vals, kth_idx = logits.topk(top_k, dim=-1)
    sample_dist = torch.distributions.categorical.Categorical(logits=kth_vals)

    token_idx_new = kth_idx.gather(dim=1, index=sample_dist.sample().unsqueeze(-1)).squeeze(-1)  
    return token_idx_new.tolist()  


def add_prefix_space(s):
    toks = s.split()

    if len(toks) < 1:
        return True

    if toks[0] in NO_PREFIX_SPACE_TOKEN_SET:
        return False
    else:
        return True


def formatted_response(stimulus, answers):
    formatted_str = ''
    tokens = stimulus.split('%')
    for i in range(len(tokens)):
        if i % 2 == 0:
            if tokens[i] == '':
                continue
            if i == 0:
                formatted_str += tokens[i].strip()
            elif i == len(tokens)-1 and tokens[i].strip() == '.':
                formatted_str += tokens[i].strip()
            else:
                if add_prefix_space(tokens[i]):
                    formatted_str += ' '+tokens[i].strip()
                else:
                    formatted_str += tokens[i].strip()
        else:
            j = i // 2
            if add_prefix_space(answers[j]):
                formatted_str += ' ' + answers[j]
            else:
                formatted_str += answers[j]
    return formatted_str.strip()


if __name__ == "__main__":
    # test add_gaps function 

    path = 'data/test.txt'
    test_add_gaps(path, '[BLANK]', '[FILLER]')
    print()

    np.random.seed(100)
    test_add_gaps(path, '[BLANK]', '[FILLER]')
    print()

    np.random.seed(100)
    test_add_gaps(path, None, None, special_format='T5')
