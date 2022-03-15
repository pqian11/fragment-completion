import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'


rs = pickle.load(open("data/exp0/human_rating_data.pkl", "rb"))


models = ['human', 'gibbscomplete', 't5-finetune', 'bart-finetune', 'ilm', 
          'gibbscomplete-nyt-lg', 't5-nyt-lg', 'bart-nyt-lg', 'ilm-nyt-lg',
          'gibbscomplete-bllip-lg', 't5-bllip-lg', 'bart-bllip-lg', 'ilm-bllip-lg'
         ]
judgment_cats = ['Grammaticality', 'Coherence', 'Interestingness', 'Overall quality']
n_stimulus = 30

for i, model in enumerate(models):
    model_summary_rs = {}
    for judgment_cat in judgment_cats:
        judgment_list = [np.mean(rs[model][stimulus_index][judgment_cat]) for stimulus_index in range(n_stimulus)]
        mean_judgment = np.mean(judgment_list) 
        print(model, judgment_cat, mean_judgment)


marker_shape_dict = {'fine-tune':'o', 'from scratch nyt':'^', 'from scratch bllip':'s'}
model_cond2name = {'fine-tune':'Pretrain/Fine-tune', 'from scratch nyt':'From scratch (NYT)', 'from scratch bllip':'From scratch (BLLIP)'}
model2color = dict(zip(models, ['k', 'C0', 'C1', 'C2', 'C3', 'C0', 'C1', 'C2', 'C3', 'C0', 'C1', 'C2', 'C3']))
model2name = dict(zip(models, ['Human', 'GibbsComplete', 'InfillT5', 'InfillBART', 'ILM', 
                               'GibbsComplete', 'InfillT5', 'InfillBART', 'ILM',
                              'GibbsComplete', 'InfillT5', 'InfillBART', 'ILM']))

plt.figure(figsize=(11,2.2))
ax = plt.gca()

for i, model in enumerate(models):
    
    if model.endswith('nyt-lg'):
        marker_shape = '^'
    elif model.endswith('bllip-lg'):
        marker_shape = 's'
    elif model == 'human':
        marker_shape = '_'
    else:
        marker_shape = 'o'
    
    model_summary_rs = {}
    judgment_list_by_cat = []
    judgment_mean_by_cat = []
    judgment_err_by_cat = []
    for j, judgment_cat in enumerate(judgment_cats):
        if model == 'human':
            judgment_list = [np.mean(rs[model][stimulus_index][judgment_cat][-13:]) for stimulus_index in range(n_stimulus)]
        elif model.endswith('nyt-lg'):
            judgment_list = [np.mean(rs[model][stimulus_index][judgment_cat][:]) for stimulus_index in range(n_stimulus)]
        elif model.endswith('bllip-lg'):
            judgment_list = [np.mean(rs[model][stimulus_index][judgment_cat][:]) for stimulus_index in range(n_stimulus)]
        else:
            judgment_list = [np.mean(rs[model][stimulus_index][judgment_cat][-4:]) for stimulus_index in range(n_stimulus)]
        judgment_mean = np.mean(judgment_list) 
#         print(len(judgment_list))
        judgment_err = 1.96*(np.std(judgment_list)/np.sqrt(len(judgment_list)))
#         print(model, judgment_cat, mean_judgment)

        judgment_list_by_cat.append(judgment_list)
        judgment_mean_by_cat.append(judgment_mean)
        judgment_err_by_cat.append(judgment_err)
        
        xs_jitter = j*3.2+i*0.2 + np.random.random(30)*0.02
        
        plt.plot(xs_jitter, judgment_list, marker_shape, mfc='none', alpha=0.15, markersize=5, c=model2color[model])


    plt.errorbar([j*3.2+i*0.2 for j in range(len(judgment_cats))], judgment_mean_by_cat, yerr=judgment_err_by_cat, 
                 fmt=marker_shape, c=model2color[model], label=model2name[model])
plt.ylim(0,101)
plt.xlim(-0.2, j*3.2+i*0.2 + 0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 10))
ax.set_xticks([j*3.2+0.2*6 for j in range(len(judgment_cats))])
ax.set_xticklabels(judgment_cats, fontsize=12)
ax.set_ylabel('Rating')
ax.xaxis.set_tick_params(length=0)

color_legend = plt.legend(handles=[Patch(facecolor=model2color[m], edgecolor=model2color[m], label=model2name[m]) for m in models[:5]], 
                        loc='upper left', bbox_to_anchor=(0.05, -0.2), ncol=3)
shape_legend = plt.legend(handles=[Line2D([0], [0], marker=marker_shape_dict[m], ls='none', color='gray', label=model_cond2name[m]) for m in ['fine-tune', 'from scratch nyt', 'from scratch bllip']], 
                        loc='upper left', bbox_to_anchor=(0.5, -0.2), ncol=2)

ax.add_artist(color_legend)
ax.add_artist(shape_legend)

plt.savefig('fig/exp0/exp0_human_judgment_summary_all.pdf', bbox_inches='tight')

plt.show()