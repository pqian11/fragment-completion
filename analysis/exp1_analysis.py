import argparse
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
matplotlib.rcParams['font.family'] = 'Arial'


def get_conf_int_stats(obs_count, total_count, method='jeffreys'):
    pref_value = obs_count/total_count
    ci_lower, ci_upper = proportion_confint(obs_count, total_count, alpha=0.05, method=method)
    return pref_value, [ci_lower, ci_upper]


def plot_rs_by_test_suite_grid_5_by_6(rs, models, human_data, test_names, model2run_indice, model2color, model2name, add_test_name=True, savepath=None):
    # Plot results as bar graph grid 5*6
    n_row = 5
    n_col = 6

    bar_width = 0.75

    fig, axs = plt.subplots(n_row, n_col, figsize=(8, 6.5), sharey='row', sharex='col')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for k, test_name in enumerate(test_names):

        row_id = k // n_col
        col_id = k % n_col

        axs[row_id, col_id].set_title('Test {}'.format(k+1), fontsize=12)
        axs[row_id, col_id].set_ylim(0,1)
        axs[row_id, col_id].set_xlim(-1.75,len(models)-0.25)
        axs[row_id, col_id].set_xticks(np.arange(0, len(models)))
        axs[row_id, col_id].set_yticks(np.arange(0, 1.2, 0.25))
        axs[row_id, col_id].set_xticklabels([])
        axs[row_id, col_id].spines['right'].set_visible(False)
        axs[row_id, col_id].spines['top'].set_visible(False)
        axs[row_id, col_id].grid(linestyle='--', alpha=0.5, zorder=0, axis='y')
        axs[row_id, col_id].set_axisbelow(True)
        
        axs[row_id, col_id].errorbar(-1, human_data[test_name]['acc_value'], yerr=[[human_data[test_name]['acc_value'] - human_data[test_name]['acc_lower']], [human_data[test_name]['acc_upper'] - human_data[test_name]['acc_value']]], label='Human', color='black', marker='None', linestyle='none')
        axs[row_id, col_id].bar(-1, human_data[test_name]['acc_value'], label='Human', width=bar_width, color='white', edgecolor='k')

        for i, model in enumerate(models):
            data = np.array([rs[model][run_index][test_name]['item_acc_list'] for run_index in model2run_indice[model]], dtype='float')
            score_averaged_across_run = np.mean(data, axis=0)
            y_mean = np.mean(score_averaged_across_run)
            yerr = 1.96*(np.std(score_averaged_across_run)/np.sqrt(len(score_averaged_across_run)))
            axs[row_id, col_id].bar(i, y_mean, label=model, width=bar_width, color=model2color[model], yerr=yerr)

    for index in range(k+1, n_row*n_col):
        row_id = index // n_col
        col_id = index % n_col
        axs[row_id, col_id].set_axis_off()

    ax = axs[4, 5]
    ax.bar(0, 0, label='Human', width=0.35, color='black', fill=False)
    for i, model in enumerate(models):
        ax.bar(i+1, 0, label=model2name[model], width=0.35, color=model2color[model])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc = 'center', bbox_to_anchor=(-1.2, 0.5), ncol=2, fontsize=12)

    fig.text(0.06, 0.5, 'Test Accuracy Score', ha='center', va='center', rotation='vertical')

    if add_test_name:
        textstr = '\n'.join(['({}) {}'.format(k+1, test_name2pretty_name[test_name]) for k, test_name in enumerate(test_names)])
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5, ec='lightgray')
        fig.text(0.94, 0.5, textstr, fontsize=10,
                verticalalignment='center', bbox=props, linespacing = 1.65)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_rs_by_test_suite_grid_3_by_9(rs, models, human_data, test_names, model2run_indice, model2color, model2name, savepath=None):
    # Plot results as bar graph grid 3*9
    n_row = 3
    n_col = 9

    bar_width = 0.75

    fig, axs = plt.subplots(n_row, n_col, figsize=(11, 3.6), sharey='row', sharex='col')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for k, test_name in enumerate(test_names):

        row_id = k // n_col
        col_id = k % n_col

        axs[row_id, col_id].set_title('Test {}'.format(k+1), fontsize=10)
        axs[row_id, col_id].set_ylim(0,1)
        axs[row_id, col_id].set_xlim(-1.75,len(models)-0.25)
        axs[row_id, col_id].spines['right'].set_visible(False)
        axs[row_id, col_id].spines['top'].set_visible(False)
        axs[row_id, col_id].grid(linestyle='--', alpha=0.5, zorder=0, axis='y')
        axs[row_id, col_id].set_xticks(np.arange(0, len(models)))
        axs[row_id, col_id].set_yticks(np.arange(0, 1.2, 0.25))
        axs[row_id, col_id].set_xticklabels([])
        axs[row_id, col_id].set_axisbelow(True)
        
        axs[row_id, col_id].errorbar(-1, human_data[test_name]['acc_value'], yerr=[[human_data[test_name]['acc_value'] - human_data[test_name]['acc_lower']], [human_data[test_name]['acc_upper'] - human_data[test_name]['acc_value']]], color='black', marker='None', linestyle='none')
        axs[row_id, col_id].bar(-1, human_data[test_name]['acc_value'], label='Human', width=bar_width, color='white', edgecolor='k')


        for i, model in enumerate(models):
            data = np.array([rs[model][run_index][test_name]['item_acc_list'] for run_index in model2run_indice[model]], dtype='float')
            score_averaged_across_run = np.mean(data, axis=0)
            y_mean = np.mean(score_averaged_across_run)
            yerr = 1.96*(np.std(score_averaged_across_run)/np.sqrt(len(score_averaged_across_run)))

            # bar plot
            axs[row_id, col_id].bar(i, y_mean, label=model2name[model], width=bar_width, color=model2color[model], yerr=yerr)


        if k == 22:
            axs[row_id, col_id].legend(loc='center', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=10)

    for index in range(k+1, n_row*n_col):
        row_id = index // n_col
        col_id = index % n_col
        axs[row_id, col_id].set_axis_off()

    fig.text(0.08, 0.5, 'Test Accuracy Score', ha='center', va='center', rotation='vertical')

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_aggregated_rs(rs, models, human_data, test_names, model2run_indice, model2color, model2name, savepath=None):
    # Plot averaged performance over all the test suites
    plt.figure(figsize=(2.5,2.5))
    ax = plt.gca()
    bar_width = 0.75

    # Use asymptotic confidence interval
    human_acc_by_test_suite = [human_data[test_name]['acc_value'] for test_name in test_names]
    human_acc_mean = np.mean(human_acc_by_test_suite)
    yerr = 1.96*(np.std(human_acc_by_test_suite)/np.sqrt(len(human_acc_by_test_suite)))
    ax.bar(-1, human_acc_mean, label='Human', width=bar_width, color='black', fill=False, yerr=yerr)
    print('Human average acc: {}'.format(human_acc_mean))

    for i, model in enumerate(models):
        data = [[rs[model][run_index][test_name]['acc'] for test_name in test_names] for run_index in model2run_indice[model]]
        test_suite_acc_list_averaged_across_run = np.mean(data, axis=0)
        mean_test_suite_acc = np.mean(test_suite_acc_list_averaged_across_run)
        yerr = 1.96*(np.std(test_suite_acc_list_averaged_across_run)/np.sqrt(len(test_suite_acc_list_averaged_across_run)))
        ax.bar(i, mean_test_suite_acc, label=model2name[model], width=bar_width, color=model2color[model], yerr=yerr)
        
    ax.set_ylim(0,1)
    ax.set_xlim(-1.75,len(models)-0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(-1, len(models)))
    ax.set_yticks(np.arange(0, 1.2, 0.25))
    ax.set_xticklabels([])

    plt.ylabel('Accuracy Score')
    plt.legend(loc = 'center', bbox_to_anchor=(1.45, 0.5))

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_summmary_across_model_conditions(exp_data_all, model_conditions, savepath=None):
    fig = plt.figure(constrained_layout=False, figsize=(7.2,2.4))

    hatch_style_list = [{'hatch':None}, {'hatch':'///'}, {'hatch':'.'}]
    model_condition2style = dict(zip(['finetune', 'nyt_from_scratch', 'bllip_from_scratch'], hatch_style_list))

    gs = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[0.25, 0.8, 0.8, 0.8], wspace=0.1)
    bar_width = 0.75

    ax = fig.add_subplot(gs[0])
    human_acc_by_test_suite = [human_data[test_name]['acc_value'] for test_name in test_names]
    human_acc_mean = np.mean(human_acc_by_test_suite)
    yerr = 1.96*(np.std(human_acc_by_test_suite)/np.sqrt(len(human_acc_by_test_suite)))
    ax.bar(0, human_acc_mean, label='Human', width=bar_width, color='black', fill=False, yerr=yerr)
    print('Human average acc: {}'.format(human_acc_mean))
    ax.set_ylim(0,1)
    ax.set_xlim(-0.75,0.75)
    ax.set_yticks(np.arange(0, 1.2, 0.25))
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels([])

    for model_cond_idx, model_condition in enumerate(model_conditions):
        ax = fig.add_subplot(gs[model_cond_idx+1])
        rs, models, model2run_indice, model2name, model2color = exp_data_all[model_condition]



        for i, model in enumerate(models):
            data = [[rs[model][run_index][test_name]['acc'] for test_name in test_names] for run_index in model2run_indice[model]]
            test_suite_acc_list_averaged_across_run = np.mean(data, axis=0)
            mean_test_suite_acc = np.mean(test_suite_acc_list_averaged_across_run)
            yerr = 1.96*(np.std(test_suite_acc_list_averaged_across_run)/np.sqrt(len(test_suite_acc_list_averaged_across_run)))
            ax.bar(i, mean_test_suite_acc, label=model2name[model], width=bar_width, color=model2color[model], yerr=yerr, **model_condition2style[model_condition])

        ax.set_ylim(0,1)
        ax.set_xlim(-0.75,len(models)-0.25)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        if model_cond_idx == 2:
            colors =['C{}'.format(k) for k in range(4)]
            model_names = ['GibbsComplete', 'InfillT5', 'InfillBART', 'ILM']
            model_condition_names = ['Pretrain/Fine-tune', 'From scratch (NYT)', 'From scratch (BLLIP)']
            color_legend = plt.legend(handles=[mpatches.Patch(facecolor='white', edgecolor='k', label='Human')]+[mpatches.Patch(facecolor=colors[k], edgecolor=colors[k], label=model_names[k]) for k in range(len(model_names))], loc='upper left', bbox_to_anchor=(1.15, 1.05), ncol=1, fontsize=10)
            hatch_legend = plt.legend(handles=[mpatches.Patch(facecolor='lightgray', edgecolor='k', linewidth=0, label=model_condition_names[k], **hatch_style_list[k]) for k in range(len(hatch_style_list))], loc='upper left', bbox_to_anchor=(1.15, 0.41), ncol=1,  fontsize=10)

            ax.add_artist(color_legend)
            ax.add_artist(hatch_legend)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def run_paired_t_tests(exp_data_all, model_conditions):
    for model_cond_idx, model_condition in enumerate(model_conditions):
        rs, models, model2run_indice, model2name, model2color = exp_data_all[model_condition]    

        model_acc_list_all = []

        for i, model in enumerate(models):
            data = [[rs[model][run_index][test_name]['acc'] for test_name in test_names] for run_index in model2run_indice[model]]  
            model_acc_list_all.append(np.mean(data, axis=0))

        print('{:<22} {:<15} {:<15} {:<6}  {:<6}'.format('Learning setup', 'Model name', 'Model name', 't_stat', 'p_value')) 
        print('-'*70)
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                d1 = np.array(model_acc_list_all[i])
                d2 = np.array(model_acc_list_all[j])
                t_stat, p_value = scipy.stats.ttest_rel(d1, d2, alternative='two-sided')
                print('{:<22} {:<15} {:<15} {:<6.3f}  {:<6.3f}'.format(model_condition, model2name[models[i]], model2name[models[j]], t_stat, p_value))            

        for i in range(len(models)):
            d1 = np.array(model_acc_list_all[i])
            d2 = [human_data[test_name]['acc_value'] for test_name in test_names]
            t_stat, p_value = scipy.stats.ttest_rel(d1, d2, alternative='two-sided')
            print('{:<22} {:<15} {:<15} {:<6.3f}  {:<6.3f}'.format(model_condition, model2name[models[i]], 'Human', t_stat, p_value))             
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze results in Evaluation III.')
    parser.add_argument('--rerank', action='store_true', help='Plot results from directly specialized models with reranking.')
    args = parser.parse_args()

    DO_RERANK='rerank' if args.rerank else 'norerank'

    DATA_DIR='data/exp1'

    test_names = ["agreement_subj", "agreement_subj-long", "agreement_emb-subj-long", "agreement_subj-with-coord", "agreement_subj-with-PP",
                    "clause_VP","clause_VP-with-PP-adjunct", "clause_VP-with-adjunct-long",
                    "clause_VP-with-complement", "clause_VP-with-complement-long", "clause_VP-gerund", 
                    "clause_phrasal-verb", "clause_phrasal-verb-with-subj", 
                    "clause_resultative", "clause_resultative-long",
                    "coord_S", "coord_VP", "coord_emb-NP", "coord_emb-VP",
                    "coord_either", "coord_neither", "coord_gap-NP", "gap_adjunct", "gap_obj", "gap_subj", "gap_phrasal-verb"]

    pretty_test_names = ["Number Agreement", "Number Agreement (Long Subject)", "Number Agreement (Embedded Clause)",
                        "Number Agreement (Coordination)", "Number Agreement (with PP)", "Clausal Structure", "Clausal Structure (PP Adjunct)",
                        "Clausal Structure (Long Adjunct)", "Clausal Structure (Complement)", "Clausal Structure (Long Complement)",
                        "Gerund", "Phrasal Verb", "Phrasal Verb (with NP)", "Resultative", "Resultative (Long NP)", "S Coordiation",
                        "VP Coordination", "Embedded NP Coordination", "Embedded VP Coordination", "Coordination (either)",
                        "Coordination (neither)", "Coordination in wh-clause", "Filler-Gap (Adjunct)", "Filler-Gap (Object)",
                        "Filler-Gap (Subject)", "Filler-Gap (Phrasal Verb)"]

    test_name2pretty_name = dict(zip(test_names, pretty_test_names))

    stimuli_example = {}

    for test_name in test_names:
        stimuli_path = '../stimuli/exp1/{}.txt'.format(test_name)
        with open(stimuli_path) as f:
            line = f.readline()
        stimuli_example[test_name] = line.strip().replace('%%', '____')

    # Load human behavioral results
    with open('{}/results/human_eval_rs.txt'.format(DATA_DIR)) as f:
        lines = f.readlines()

    lines = [line.strip().split() for line in lines if line.strip() != '']
    human_data = {}
    for line in lines:
        test_name = line[1]
        human_data[test_name] = {}
        human_data[test_name]['acc'] = float(line[2]) 
        proportions1 = [float(item) for item in line[3].split('/')]
        proportions2 = [float(item) for item in line[4].split('/')]

        acc_value, [acc_lower, acc_upper] = get_conf_int_stats(proportions1[0] + proportions2[0], proportions1[1] + proportions2[1], method='jeffreys')

        human_data[test_name]['acc_value'] = acc_value
        human_data[test_name]['acc_lower'] = acc_lower
        human_data[test_name]['acc_upper'] = acc_upper

    exp_data_all = {}

    fig_dir = 'fig/exp1/'

    model_name_list = ['GibbsComplete', 'InfillT5', 'InfillBART', 'ILM']
    model_color_list = ['C0', 'C1', 'C2', 'C3']

    model_conditions = ['finetune', 'nyt_from_scratch', 'bllip_from_scratch']

    model_condition2dir_name = dict(zip(model_conditions, ['pretrain-finetune', 'nyt-lg', 'bllip-lg']))

    for model_condition in model_conditions:

        if model_condition == 'nyt_from_scratch':
            # Load and visualize results for models trained from scratch on a subset of NYT
            models = ['gibbscomplete-nyt-lg', 't5-nyt-lg', 'bart-nyt-lg', 'ilm-nyt-lg']
            model2run_indice = {'gibbscomplete-nyt-lg':['0001', '0002', '0003'], 't5-nyt-lg':['0001', '0002', '0003'],
                                'bart-nyt-lg':['0001', '0002', '0003'], 'ilm-nyt-lg':['0001', '0002', '0003']}
        elif model_condition == 'finetune':
            # Load and visualize results for pretrained models finetuned on a subset of NYT 2007
            models = ['gibbscomplete', 't5-finetune', 'bart-finetune', 'ilm']

            if DO_RERANK == 'rerank':
                model2run_indice = {'gibbscomplete':['0001', '0002', '0003'], 't5-finetune':['1001', '1002', '1003'],
                                'bart-finetune':['1001', '1002', '1003'], 'ilm':['1001', '1002', '1003']}
            else:
                model2run_indice = {'gibbscomplete':['0001', '0002', '0003'], 't5-finetune':['0001', '0002', '0003'],
                    'bart-finetune':['0001', '0002', '0003'], 'ilm':['0001', '0002', '0003']}

        elif model_condition == 'bllip_from_scratch':
            # Load and visualize results for models trained from scratch on BLLIP-lg
            models = ['gibbscomplete-bllip-lg', 't5-bllip-lg', 'bart-bllip-lg', 'ilm-bllip-lg']
            model2run_indice = {'gibbscomplete-bllip-lg':['0101', '0102', '0103'], 't5-bllip-lg':['0001', '0002', '0003'],
                                'bart-bllip-lg':['0001', '0002', '0003'], 'ilm-bllip-lg':['0001', '0002', '0003']}            

        model2name = dict(zip(models, model_name_list))
        model2color = dict(zip(models, model_color_list))
        

        rs = {}
        for model in models:
            rs[model] = {}
            for run_index in model2run_indice[model]:
                rs[model][run_index] = {}
                for test_name in test_names:
                    rs[model][run_index][test_name] = {'acc':None, 'item_acc_list':[]}

                if model.startswith('gibbscomplete'):
                    path = '{}/results/{}/{}_{}_eval_rs.txt'.format(DATA_DIR, model_condition2dir_name[model_condition], model, run_index)
                else:
                    if DO_RERANK == 'rerank':
                        path = '{}/results/{}/{}_rerank_{}_eval_rs.txt'.format(DATA_DIR, model_condition2dir_name[model_condition], model, run_index)
                    else:
                        path = '{}/results/{}/{}_{}_eval_rs.txt'.format(DATA_DIR, model_condition2dir_name[model_condition], model, run_index)
                lines = open(path).readlines()
                lines = [line.strip().split() for line in lines]
                for line in lines:
                    if len(line) < 1:
                        continue
                    test_name = line[0]
                    item_acc = float(line[2])
                    rs[model][run_index][test_name]['item_acc_list'].append(item_acc)

                for test_name in test_names:
                    rs[model][run_index][test_name]['acc'] = np.mean(rs[model][run_index][test_name]['item_acc_list'])

        plot_rs_by_test_suite_grid_5_by_6(rs, models, human_data, test_names, model2run_indice, model2color, model2name, savepath='{}/exp1_{}_{}_eval_grid_bar_5x6.pdf'.format(fig_dir, DO_RERANK, model_condition))
        plot_rs_by_test_suite_grid_3_by_9(rs, models, human_data, test_names, model2run_indice, model2color, model2name, savepath='{}/exp1_{}_{}_eval_grid_bar_3x9.pdf'.format(fig_dir, DO_RERANK, model_condition))
        # plot_aggregated_rs(rs, models, human_data, test_names, model2run_indice, model2color, model2name, savepath='{}/exp1_{}_{}_eval_bar_average_score.pdf'.format(fig_dir, model_condition, DO_RERANK))

        exp_data_all[model_condition] = [rs, models, model2run_indice, model2name, model2color]

    run_paired_t_tests(exp_data_all, model_conditions)

    plot_summmary_across_model_conditions(exp_data_all, model_conditions, savepath='{}/exp1_{}_overall_summary.pdf'.format(fig_dir, DO_RERANK))

