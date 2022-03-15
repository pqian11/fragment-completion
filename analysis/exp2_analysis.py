import os
import argparse
import json
import pickle
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import *
import nltk
from nltk.tree import Tree
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib import colors
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'


def load_fdist_from_str(line):
    """
    Load a frequency dictionary from a line of identifier, frequecy pairs,
    e.g. S 10 NP 5 VP 5 ...
    """
    fdist = {}
    tokens = line.strip().split()
    nPair = int(len(tokens)/2)
    assert len(tokens) % 2 == 0
    for i in range(nPair):
        identifier = tokens[i*2]
        freq = float(tokens[i*2+1])
        fdist[identifier] = freq
    return fdist


def load_pair_props(path, stimuli_set):
    pair_dict = {}

    with open(path, encoding='utf-8') as f:
        for line in f:
            pairs_chunk, freq_chunk, fdist_chunk = line.strip().split('\t')
            pair_tokens = pairs_chunk.split()

            if '_'.join(pair_tokens[:2]) not in stimuli_set:
                continue

            fdist = load_fdist_from_str(fdist_chunk)
            pair = ' '.join(pair_tokens[:2])
            if pair not in pair_dict:
                pair_dict[pair] = fdist
            else:
                for k in fdist.keys():
                    if k in pair_dict[pair]:
                        pair_dict[pair][k] += fdist[k]
                    else:
                        pair_dict[pair][k] = fdist[k]
    return pair_dict


def get_target_indice(targets, answers, sent):
    """
    Find the indice of the target words in the sentence,
    given the sequence of fragments and answers to the blanks.
    """
    prefix = ''
    b0 = ''.join(answers[0].split())
    b1 = b0 + targets[0] + ''.join(answers[1].split())
    index0 = -1
    index1 = -1
    for k, w in enumerate(sent):
        prefix += w
        if prefix == b0:
            index0 = k + 1
        elif prefix == b1:
            index1 = k + 1

    assert index0 != -1 
    assert index1 != -1 
    return index0, index1


def get_target_indice_from_leaves(targets, sent):
    """
    Find the indice of the target words in the terminal sequence of a parsed sentence.
    """
    target_indice = []
    idx = 0
    current_target = targets[idx]
    for k, w in enumerate(sent):
        if w == current_target:
            target_indice.append(k)
            idx += 1
            if idx == len(targets):
                break
            current_target = targets[idx]
    try:
        len(target_indice) == len(targets)
    except ValueError:
        print(targets)
        print(' '.join(sent))

    return target_indice  


def get_lca_label(ptree, leaf_index1, leaf_index2):
    """
    Find the lowest common ancestor node given the indices of two leaf nodes
    """
    location1 = ptree.leaf_treeposition(leaf_index1)
    location2 = ptree.leaf_treeposition(leaf_index2)

    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i += 1
    lca_label = ptree[location1[:i]].label()
    return lca_label


def get_CI_stats(obs_count, total_count, method='jeffreys'):
    pref_value = obs_count/total_count
    ci_lower, ci_upper = proportion_confint(obs_count, total_count, alpha=0.05, method=method)
    return pref_value, [ci_lower, ci_upper]


def update_lca_pref(prefDict, lca_cat, pref_val, pref_ci):
    prefDict['{}_val'.format(lca_cat)] = pref_val
    prefDict['{}_ci'.format(lca_cat)] = pref_ci
    return  


def update_prefs(prefDict, lca_cats, fdist, total_count):
    prefDict['total_count'] = total_count

    target_lca_cum_count = 0

    for lca_cat in lca_cats:
        lca_cat_count = fdist[lca_cat] if lca_cat in fdist else 0
        pref = lca_cat_count/total_count
        prefDict[lca_cat] = pref
        pref_val, pref_ci = get_CI_stats(lca_cat_count, total_count)
        update_lca_pref(prefDict, lca_cat, pref_val, pref_ci)
        target_lca_cum_count += lca_cat_count
    
    other_lca_count = total_count - target_lca_cum_count
    other_pref = other_lca_count/total_count
    prefDict['other'] = other_pref
    other_pref_val, other_pref_ci = get_CI_stats(other_lca_count, total_count)
    update_lca_pref(prefDict, 'other', other_pref_val, other_pref_ci)
    return


def compute_mse_all_by_stimulus(prefs, lca_cats, tag_cats, return_all=False):
    mse_all = []

    for tag_cat in tag_cats:
        for group in stimuli[tag_cat]:
            xs = [prefs['model'][tag_cat][group]['{}_val'.format(lca_cat)] for lca_cat in lca_cats]
            ys = [prefs['human'][tag_cat][group]['{}_val'.format(lca_cat)] for lca_cat in lca_cats]

            group_mse = np.mean((np.array(xs) - np.array(ys))**2)
            mse_all.append(group_mse)

    mse = np.mean(mse_all)

    if return_all:
        return mse, mse_all
    else:
        return mse


def plot_mse_bar_summary_across_model_condition(exp_data_all, model_conditions, savepath=None):
    fig = plt.figure(constrained_layout=False, figsize=(10*1,4.5*1))

    tag_cats = ['N_N', 'J_J', 'J_N']
    lca_cat_targets = ['S', 'NP', 'VP', 'ADJP', 'other']

    tag_cat2style = {'N_N':{'color':plt.cm.Set2(0)}, 'J_J':{'color':plt.cm.Set2(1)}, 'J_N':{'color':plt.cm.Set2(2)}}
    tag_cat2label = {'N_N':'Noun Noun', 'J_J':'Adj Adj', 'J_N':'Adj Noun'}

    hatch_style_list = [{'hatch':None}, {'hatch':'///'}, {'hatch':'.'}]
    model_condition2style = dict(zip(['finetune', 'nyt_from_scratch', 'bllip_from_scratch'], hatch_style_list))

    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1, 2.5, 2.5], height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)

    for model_cond_idx, model_condition in enumerate(model_conditions):
        prefs_all, models, run_indice_dict, model2name, model2color = exp_data_all[model_condition]

        for column_idx in range(3):
            ax = fig.add_subplot(gs[model_cond_idx, column_idx])

            # Plot overall summary
            if column_idx == 0:
                ax.set_axisbelow(True)
                bar_width = 0.75

                mse_list_all = []
                for i, model in enumerate(models):
                    mse_all_list = []
                    for run_index in run_indice_dict[model]:
                        _, mse_all = compute_mse_all_by_stimulus(prefs_all[model][run_index], lca_cat_targets, tag_cats, return_all=True)
                        mse_all_list.append(mse_all)
                    mse_list = np.mean(mse_all_list, axis=0)
                    mean_score = np.mean(mse_list)
                    mean_score_err = (np.std(mse_list)/np.sqrt(len(mse_list)))*1.96 

                    ax.bar(i, mean_score, label=model2name[model], width=bar_width, yerr=mean_score_err, error_kw=dict(lw=1.2), color=model2color[model], **model_condition2style[model_condition])

                    mse_list_all.append(list(mse_list))

                    # print(i, model, len(mse_list))

                mse_list_all = np.array(mse_list_all)
                # print(mse_list_all.shape)
                # print(len(mse_list_all))

                mse_list_averaged_across_model = np.mean(mse_list_all, axis=0)

                print('{:<20} {:<6} {:<15} {:<15} {:<6}  {:<6}'.format('Learning setup', 'Group', 'Model1', 'Model2', 't_stat', 'p_value')) 
                print('-'*75)
                for i in range(len(models)):
                    for j in range(i+1, len(models)):
                        d1 = np.array(mse_list_all[i])
                        d2 = np.array(mse_list_all[j])
                        t_stat, p_value = scipy.stats.ttest_rel(d1, d2, alternative='two-sided')
                        print('{:<20} {:<6} {:<15} {:<15} {:<6.3f}  {:<6.3f}'.format(model_condition, 'n/a', model2name[models[i]], model2name[models[j]], t_stat, p_value))


                ax.set_xlim(0.75,len(models)-0.25)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xticks(np.arange(-1, len(models)))
                ax.xaxis.set_tick_params(length=0)
                ax.set_xticklabels([])
                ax.set_yticks(np.arange(0, 0.045, 0.01))
                ax.set_ylabel('Mean Squared Error', fontsize=8)

                label_x_position = -0.65
                label_y_position = 0.5
                if model_cond_idx == 0:
                    ax.text(label_x_position, label_y_position, 'Pre-train/Fine-tune', rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=8.5, transform=ax.transAxes)
                elif model_cond_idx == 1:
                    ax.text(label_x_position, label_y_position, 'From scratch (NYT)', rotation='vertical', horizontalalignment='center', verticalalignment='center',  fontsize=8.5, transform=ax.transAxes)
                elif model_cond_idx == 2:
                    ax.text(label_x_position, label_y_position, 'From scratch (BLLIP)', rotation='vertical', horizontalalignment='center', verticalalignment='center',  fontsize=8.5, transform=ax.transAxes)
                else:
                    raise NotImplementedError

                if model_cond_idx  == 0:
                    ax.set_title('Aggregated Mean Squared Error', fontsize=10)

            # Plot summary by LCA category
            elif column_idx == 1:
                bar_width = 0.15
                mse_list_by_model_lca_cat_all = [[] for _ in range(len(models))]

                # Plot aggregated results from multiple random runs; average random runs first
                for i, model in enumerate(models):
                    for j, lca_cat in enumerate(lca_cat_targets):
                        mse_all_list = []
                        for run_index in run_indice_dict[model]:
                            _, mse_all = compute_mse_all_by_stimulus(prefs_all[model][run_index], [lca_cat], tag_cats, return_all=True)
                            mse_all_list.append(mse_all)
                        mse_list = np.mean(mse_all_list, axis=0)
                        mse_mean = np.mean(mse_list)
                        mse_mean_err = (np.std(mse_list)/np.sqrt(len(mse_list)))*1.96    
                        ax.bar(j+0.18*(i-len(models)/2), mse_mean, yerr=mse_mean_err, error_kw=dict(lw=0.85), label=model2name[model], width=bar_width, color=model2color[model], **model_condition2style[model_condition])                                    
             
                        mse_list_by_model_lca_cat_all[i].append(mse_list)

                # Run paired t-test
                for k in range(len(lca_cat_targets)):
                    for i in range(len(models)):
                        for j in range(i+1, len(models)):
                            d1 = np.array(mse_list_by_model_lca_cat_all[i][k])
                            d2 = np.array(mse_list_by_model_lca_cat_all[j][k])
                            t_stat, p_value = scipy.stats.ttest_rel(d1, d2, alternative='two-sided')
                            print('{:<20} {:<6} {:<15} {:<15} {:<6.3f}  {:<6.3f}'.format(model_condition, lca_cat_targets[k], model2name[models[i]], model2name[models[j]], t_stat, p_value))

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xticks(np.arange(0, len(lca_cat_targets)) + 0.18*(1.5-len(models)/2))
                ax.set_xticklabels(lca_cat_targets)
                ax.set_yticks(np.arange(0, 0.085, 0.02))
                ax.xaxis.set_tick_params(length=0)

                if model_cond_idx  == 0:
                    ax.set_title('Mean Squared Error by LCA Category', fontsize=10)

            # Plot summary by stimulus condition
            elif column_idx == 2:
                bar_width = 0.08
                tag_cat_labels = [tag_cat2label[tag_cat] for tag_cat in tag_cats]
                mse_list_by_model_tag_cat_all = [[] for _ in range(len(models))]

                # Plot aggregated results from multiple random runs
                for i, model in enumerate(models):                    
                    for j, tag_cat in enumerate(tag_cats):
                        mse_all_list = []
                        for run_index in run_indice_dict[model]:
                            _, mse_all = compute_mse_all_by_stimulus(prefs_all[model][run_index], lca_cat_targets, [tag_cat], return_all=True)
                            mse_all_list.append(mse_all)
                        mse_list = np.mean(mse_all_list, axis=0)
                        mse_mean = np.mean(mse_list)
                        mse_mean_err = (np.std(mse_list)/np.sqrt(len(mse_list)))*1.96                                        
                        ax.bar(j+0.095*(i-len(models)/2), mse_mean, yerr=mse_mean_err, error_kw=dict(lw=0.85), label=model2name[model], width=bar_width, color=model2color[model], **model_condition2style[model_condition])

                        mse_list_by_model_tag_cat_all[i].append(mse_list)

                # Run paired t-test
                for k in range(len(tag_cats)):
                    for i in range(len(models)):
                        for j in range(i+1, len(models)):
                            d1 = np.array(mse_list_by_model_tag_cat_all[i][k])
                            d2 = np.array(mse_list_by_model_tag_cat_all[j][k])
                            t_stat, p_value = scipy.stats.ttest_rel(d1, d2, alternative='two-sided')
                            print('{:<20} {:<6} {:<15} {:<15} {:<6.3f}  {:<6.3f}'.format(model_condition, tag_cats[k], model2name[models[i]], model2name[models[j]], t_stat, p_value))
                print()

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xticks(np.arange(0, len(tag_cats)) + 0.095*(1.5-len(models)/2))
                ax.xaxis.set_tick_params(length=0)
                ax.set_xticklabels(tag_cat_labels)
                ax.set_yticks(np.arange(0, 0.045, 0.01))

                if model_cond_idx  == 0:
                    ax.set_title('Mean Squared Error by Stimulus Condition', fontsize=10)


            if model_cond_idx == 0 and column_idx == 1:
                colors =['C{}'.format(k) for k in range(4)]
                model_names = ['GibbsComplete', 'InfillT5', 'InfillBART', 'ILM']
                model_condition_names = ['Pretrain/Fine-tune', 'From scratch (NYT)', 'From scratch (BLLIP)']
                color_legend = plt.legend(handles=[mpatches.Patch(facecolor=colors[k], edgecolor=colors[k], label=model_names[k]) for k in range(len(model_names))], loc='lower center', bbox_to_anchor=(0.05, 1.25), ncol=2, fontsize=9)
                hatch_legend = plt.legend(handles=[mpatches.Patch(facecolor='lightgray', edgecolor='k', linewidth=0, label=model_condition_names[k], **hatch_style_list[k]) for k in range(len(hatch_style_list))], loc='lower center', bbox_to_anchor=(1.25, 1.25), ncol=2,  fontsize=9)

                ax.add_artist(color_legend)
                ax.add_artist(hatch_legend)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


def plot_scatter_plots_across_models_for_specific_run(prefs_all, models, run_indice_dict, model2name, idx=4, savepath=None):
    fig = plt.figure(constrained_layout=False, figsize=(9,9/4*5))

    lca_cat2style = {'S':{'marker':'o', 'ms':5}, 'NP':{'marker':'x','ms':5}, 'ADJP':{'marker':'s','ms':5}, 
                    'VP':{'marker':'^','ms':5}, 'other':{'marker':'d','ms':3.5}}
    tag_cat2style = {'N_N':{'color':plt.cm.Set2(0)}, 'J_J':{'color':plt.cm.Set2(1)}, 'J_N':{'color':plt.cm.Set2(2)}}
    tag_cat2label = {'N_N':'Noun Noun', 'J_J':'Adj Adj', 'J_N':'Adj Noun'}

    gs = fig.add_gridspec(nrows=5, ncols=4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1, 1, 1], wspace=0.1)

    for i, model in enumerate(models):
        run_index = run_indice_dict[model][int(idx)-1]

        prefs = prefs_all[model][run_index]

        for k, lca_cat in enumerate(['S', 'NP', 'VP', 'ADJP', 'other']):
            ax = fig.add_subplot(gs[k, i])
            if i == 0 and k < 4:
                ax.set_xticklabels([])
            elif k == 4 and i > 0:
                ax.set_yticklabels([])
            elif i != 0 and k != 4:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            
            if i == 0:
                ax.set_ylabel(r'Human  P'+r'({})'.format(lca_cat))

            if k == 0:
                ax.title.set_text(model2name[model])

            ax.plot([0,1], [0,1], '--', color='lightgray')

            xs_all = []
            ys_all = []

            for tag_cat in tag_cats:
                if lca_cat not in lca_cats_all[tag_cat] and (lca_cat != 'other'):
                    continue
                xs =[]
                ys = []
                xerrs = [[], []]
                yerrs = [[], []]
                for group in stimuli[tag_cat]:
                    x = prefs['model'][tag_cat][group]['{}_val'.format(lca_cat)]
                    x_lower, x_upper = prefs['model'][tag_cat][group]['{}_ci'.format(lca_cat)]

                    y = prefs['human'][tag_cat][group]['{}_val'.format(lca_cat)]
                    y_lower, y_upper = prefs['human'][tag_cat][group]['{}_ci'.format(lca_cat)]

                    xs.append(x)
                    ys.append(y)
                    xerrs[0].append(x - x_lower)
                    xerrs[1].append(x_upper - x)
                    yerrs[0].append(y - y_lower)
                    yerrs[1].append(y_upper - y)

                # _, _, bars = ax.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, **tag_cat2style[tag_cat], **lca_cat2style[lca_cat], ls='None', elinewidth=0.2)
                _, _, bars = ax.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, mec=tag_cat2style[tag_cat]['color'], ecolor=tag_cat2style[tag_cat]['color'],
                    mfc=get_lighter_color(tag_cat2style[tag_cat]['color'][:3], 0.5), **lca_cat2style[lca_cat], ls='None', elinewidth=0.2)

                xs_all += xs
                ys_all += ys

            ax.set_xlim(0,1.05)
            ax.set_ylim(0,1.05)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            rho = list(scipy.stats.spearmanr(xs_all, ys_all))[0]
            ax.annotate(r'$\rho$'+ r'$= {}$'.format(np.round(rho, 3)), xy=(0.72, 0.05), fontsize=7.5, family='arial')

            if k == 0 and i == 0:
                lca_legend_elements = []
                for lca_cat in ['S', 'NP', 'VP', 'ADJP', 'other']:
                    lca_legend_elements.append(Line2D([0], [0], marker=lca_cat2style[lca_cat]['marker'], label=lca_cat, linestyle='None', 
                        markersize=5, color=get_lighter_color(colors.to_rgba('gray')[:3], 0.5), mec='gray'))
                legend = plt.legend(handles=lca_legend_elements, loc='upper left', ncol=3, bbox_to_anchor=(0, 1.8), title='LCA Category')
                ax.add_artist(legend)
                item_legend_elements = []
                for tag_cat in ['N_N', 'J_J', 'J_N']:
                    patch = mpatches.Patch(**tag_cat2style[tag_cat], label=tag_cat2label[tag_cat])
                    item_legend_elements.append(patch)
                legend = plt.legend(handles=item_legend_elements, ncol=3, loc='upper left', bbox_to_anchor=(2, 1.8), title='Condition')
                ax.add_artist(legend)

    fig.text(0.5, 0.065, r"Model  P(LCA Category | "+r"___ $w_1$ ___ $w_2$ ___"+")", ha='center')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


def get_lighter_color(color, percent):
    """
    Assume color to be rgb format between (0, 0, 0) and (1, 1, 1)
    """
    color = np.array(color)
    color_diff = np.array([1,1,1]) - color
    return color + color_diff*percent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze results in Evaluation III.')
    parser.add_argument('--rerank', action='store_true', help='Plot results from directly specialized models with reranking.')
    parser.add_argument('--recompute', action='store_true', help='Recompute the summary statistics from data.')
    args = parser.parse_args()

    DATA_DIR='data/exp2'
    OUTPUT_DIR='model_output/exp2'

    tag_cats = ['N_N', 'J_J', 'J_N']

    # Load stimuli
    stimuli = {}
    stimuli_set = {}
    for tag_cat in tag_cats:
        with open('{}/exp2_materials_{}.txt'.format(DATA_DIR, tag_cat)) as f:
            lines = f.readlines()
        lines = ['_'.join(line.strip().split()) for line in lines]
        stimuli[tag_cat] = lines
        stimuli_set[tag_cat] = set(lines)


    # Load human behavioral data
    print('Loading behavioral data...')

    responses = {}
    for tag_cat in tag_cats:
        responses[tag_cat] = {}
        for group in stimuli[tag_cat]:
            responses[tag_cat][group] = {}
            responses[tag_cat][group]['lca_labels'] = []
            responses[tag_cat][group]['blank_lengths'] = [[], [], []]

    data = {}
    for tag_cat in tag_cats:
        data[tag_cat] = json.load(open('{}/infilling-exp2_{}_data_extracted_parsed.json'.format(DATA_DIR, tag_cat)))
        for subject_data in data[tag_cat]:
            if tag_cat == 'N_N' and subject_data['index'] in set([37, 41]):
                continue
            trials = subject_data['trials']
            for trial in trials:
                group = trial['group']
                targets = group.split('_')
                answers = trial['answers']
                tree_str = trial['tree']
                t = Tree.fromstring(tree_str)
                sent = t.leaves()
                index0, index1 = get_target_indice(targets, answers, sent)
                lca_label = get_lca_label(t, index0, index1)
                responses[tag_cat][group]['lca_labels'].append(lca_label)
                responses[tag_cat][group]['blank_lengths'][0].append(index0)
                responses[tag_cat][group]['blank_lengths'][1].append(index1-index0-1)
                responses[tag_cat][group]['blank_lengths'][2].append(len(sent)-index1-1)

    exp_data_all = {}

    DO_RERANK='rerank' if args.rerank else 'norerank'
    model_conditions = ['finetune', 'nyt_from_scratch', 'bllip_from_scratch']

    if args.recompute:
        model_condition2dir_name = dict(zip(model_conditions, ['pretrain-finetune', 'nyt-lg', 'bllip-lg']))
        model_name_list = ['GibbsComplete', 'InfillT5', 'InfillBART', 'ILM']

        for model_condition in model_conditions:

            if model_condition == 'finetune':
                models = ['gibbscomplete', 't5-finetune', 'bart-finetune', 'ilm']
                run_indice_dict = {'gibbscomplete':['0001', '0002', '0003', '0004'],
                                    't5-finetune':['1101', '1102', '1103', '1104'],
                                    'bart-finetune':['1101', '1102', '1103', '1104'],
                                    'ilm':['1101', '1102', '1103', '1104']}

            elif model_condition == 'nyt_from_scratch':
                models = ['gibbscomplete-nyt-lg', 't5-nyt-lg', 'bart-nyt-lg', 'ilm-nyt-lg']
                run_indice_dict = {'gibbscomplete-nyt-lg':['0001', '0002', '0003', '0004'],
                                    't5-nyt-lg':['0001', '0002', '0003', '0004'],
                                    'bart-nyt-lg':['0001', '0002', '0003', '0004'],
                                    'ilm-nyt-lg':['0001', '0002', '0003', '0004']}

            elif model_condition == 'bllip_from_scratch':
                models = ['gibbscomplete-bllip-lg', 't5-bllip-lg', 'bart-bllip-lg', 'ilm-bllip-lg']
                run_indice_dict = {'gibbscomplete-bllip-lg':['0101', '0102', '0103', '0104'],
                                    't5-bllip-lg':['0001', '0002', '0003', '0004'],
                                    'bart-bllip-lg':['0001', '0002', '0003', '0004'],
                                    'ilm-bllip-lg':['0001', '0002', '0003', '0004']
                                    }            
            else:
                raise NotImplementedError

            model2name = dict(zip(models, model_name_list))
            model2color = dict(zip(models, ['C0', 'C1', 'C2', 'C3']))

            prefs_all = {}

            for model in models:
                prefs_all[model] = {}
                for run_index in run_indice_dict[model]:
                    if model.startswith('gibbscomplete'):
                        model_output_path = '{output_dir}/prd/{model}/{run_index}/{model}_exp2_output.prd'.format(output_dir=OUTPUT_DIR, model=model, run_index=run_index)
                    else:
                        if model_condition == 'finetune':
                            if DO_RERANK == 'rerank':
                                model_output_path = '{output_dir}/prd/{model}/rerank/{run_index}/{model}_exp2_output.prd'.format(output_dir=OUTPUT_DIR, model=model, run_index=run_index)
                            else:
                                model_output_path = '{output_dir}/prd/{model}/{run_index}/{model}-norerank_exp2_output.prd'.format(output_dir=OUTPUT_DIR, model=model, run_index=run_index)
                        else:
                            if DO_RERANK == 'rerank':
                                model_output_path = '{output_dir}/prd/{model}/rerank/{run_index}/{model}_exp2_output.prd'.format(output_dir=OUTPUT_DIR, model=model, run_index=run_index)
                            else:
                                model_output_path = '{output_dir}/prd/{model}/{run_index}/{model}_exp2_output.prd'.format(output_dir=OUTPUT_DIR, model=model, run_index=run_index)
                    
                    # Load model output data
                    print('Loading model data from {}...'.format(model_output_path))
                    pair_props = {'N_N':{}, 'J_J':{}, 'J_N':{}}
                    n_completion = 35
                    n_line_per_block = n_completion + 2

                    lines = open(model_output_path, encoding='utf-8').readlines()
                    for k in range(len(lines)//n_line_per_block):
                        if k < 40:
                            tag_cat = 'N_N'
                        elif k < 80:
                            tag_cat = 'J_J'
                        else:
                            tag_cat = 'J_N'
                        stimulus_line = lines[k*n_line_per_block]
                        targets = stimulus_line.replace('%%', '').strip().split()[:2]
                        group = ' '.join(targets)
                        lca_labels = []

                        for j in range(1, n_completion+1):
                            tree_str = lines[k*n_line_per_block+j]
                            t = Tree.fromstring(tree_str)
                            sent = t.leaves()
                            target_indice = get_target_indice_from_leaves(targets, sent)
                            if len(target_indice) == 2: 
                                lca_label = get_lca_label(t, target_indice[0], target_indice[1])
                                lca_labels.append(lca_label)

                        pair_props[tag_cat][group] = nltk.FreqDist(lca_labels)

                    prefs = {}
                    prefs['human'] = {}
                    prefs['model'] = {}

                    lca_cats_all = {'J_N':['S', 'NP', 'VP', 'ADJP'], 'N_N':['S', 'NP', 'VP', 'ADJP'], 'J_J':['S', 'NP', 'VP', 'ADJP']}

                    for tag_cat in tag_cats:
                        lca_cats = lca_cats_all[tag_cat]

                        prefs['human'][tag_cat] = {}
                        prefs['model'][tag_cat] = {}

                        for group in stimuli[tag_cat]:
                            # compute preferences in human responses
                            prefs['human'][tag_cat][group] = {}
                            fdist = nltk.FreqDist(responses[tag_cat][group]['lca_labels'])
                            total_count = len(responses[tag_cat][group]['lca_labels'])
                            update_prefs(prefs['human'][tag_cat][group], lca_cats, fdist, total_count)

                            # compute preferences in model-generated completions
                            prefs['model'][tag_cat][group] = {}
                            pair = ' '.join(group.split('_'))
                            fdist = pair_props[tag_cat][pair]
                            total_count = np.sum([p for x, p in fdist.items()])
                            update_prefs(prefs['model'][tag_cat][group], lca_cats, fdist, total_count)
                            
                    prefs_all[model][run_index] = prefs
                             
            exp_data_all[model_condition] = [prefs_all, models, run_indice_dict, model2name, model2color]


        with open("{}/exp2_{}_eval_data_all.pkl".format(DATA_DIR, DO_RERANK), "wb") as f:
            pickle.dump(exp_data_all, f)

    assert os.path.isfile("{}/exp2_{}_eval_data_all.pkl".format(DATA_DIR, DO_RERANK)) 

    lca_cats_all = {'J_N':['S', 'NP', 'VP', 'ADJP'], 'N_N':['S', 'NP', 'VP', 'ADJP'], 'J_J':['S', 'NP', 'VP', 'ADJP']}
    with open("{}/exp2_{}_eval_data_all.pkl".format(DATA_DIR, DO_RERANK), 'rb') as f:
        exp_data_all = pickle.load(f)

    plot_mse_bar_summary_across_model_condition(exp_data_all, model_conditions, 
        savepath='fig/exp2/exp2_{}_mse_summary_all.pdf'.format(DO_RERANK))

    for model_cond_idx, model_condition in enumerate(model_conditions):
        prefs_all, models, run_indice_dict, model2name, model2color = exp_data_all[model_condition]
        plot_scatter_plots_across_models_for_specific_run(prefs_all, models, run_indice_dict, model2name, 
            idx=1, savepath='fig/exp2/exp2_{}_{}_scatter_all.pdf'.format(DO_RERANK, model_condition))

