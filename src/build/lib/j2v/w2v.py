#! /usr/bin/env python
# coding: utf-8
# author: Hao Peng
# email: pengh1992@gmail.com

from j2v import mag
from j2v import util
from bhtsne import tsne
import numpy as np
import pandas as pd
import sys
import random
from itertools import combinations
import matplotlib.pyplot as plt
import multiprocessing
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


color2hex = {'Dandelion': '#CDC673', 'Red': '#FF0000', 'Peach': '#8B6508', 'Yellow': '#ffff0c', 'Mahogany': '#FF6600', 'OliveGreen': '#20c157', 'SkyBlue': '#2dffff', 'Blue': '#0001fa', 'Emerald': '#6B8E23', 'BrickRed': '#8B0000', 'Mulberry': '#663788', 'Lavender': '#fb01ff', 'Canary': '#EE9A00', 'Black': '#000000'}

dis2color = {'Brain Research': 'Dandelion', 'Medical Specialties': 'Red', 'Health Professionals': 'Peach', 'Social Sciences': 'Yellow', 'Earth Sciences': 'Mahogany', 'Biology': 'OliveGreen', 'Chemical, Mechanical, & Civil Engineering': 'SkyBlue', 'Chemistry': 'Blue', 'Biotechnology': 'Emerald', 'Infectious Diseases': 'BrickRed', 'Math & Physics': 'Mulberry', 'Humanities': 'Canary', 'Electrical Engineering & Computer Science': 'Lavender', 'Interdiscipline': 'Black'}

# dis_name_abbr = {'Brain Research': 'Brain', 'Medical Specialties': 'Med.', 'Health Professionals': 'Health', 'Social Sciences': 'Social', 'Earth Sciences': 'Earth', 'Biology': 'Bio.', 'Chemical, Mechanical, & Civil Engineering': 'Eng.', 'Chemistry': 'Chem.', 'Biotechnology': 'Biotech.', 'Infectious Diseases': 'Infect. Dis.', 'Math & Physics': 'Math', 'Humanities': 'Hum.', 'Electrical Engineering & Computer Science': 'EE&CS'}

hex2color = dict((h, c) for (c, h) in color2hex.items())
color2dis = dict((c, d) for (d, c) in dis2color.items())

domain = {'natural science': ['Brain Research', 'Medical Specialties', 'Health Professionals', 'Earth Sciences', 'Biology', 'Chemistry', 'Biotechnology', 'Infectious Diseases', 'Math & Physics'], 'social science': ['Social Sciences', 'Humanities']}

abbr = {'Cell': 'Cell', 'Physical Review Letters': 'Phys. Rev. Lett.', 'The EMBO Journal': 'EMBO J.', \
        'Europhysics Letters (epl)': 'EPL', 'Biochemical Journal':'Biochem. J.', \
        'The Journal of Physical Chemistry': 'J. Phys. Chem.', 'Genetics and Molecular Biology': 'Genet. Mol. Biol.',\
        'Physics Letters A': 'PRA', 'Brain Cell Biology': 'Brain Cell Biol.', 'Physical Review B': 'PRB', \
        'Nature Biotechnology': 'Nat. Biotechnol.',\
        'Nanotechnology': 'Nanotechnology', 'Bioinformatics': 'Bioinformatics', 'Physical Review E': 'PRE',\
        'Current Opinion in Chemical Biology': 'Curr. Opin. Chem. Biol.', 'Nano Letters': 'Nano Lett.',\
        'Journal of Theoretical Biology': 'J. Theor. Biol.', 'Progress of Theoretical Physics': 'Progr. Theor. Phys.',\
        'Experimental Cell Research': 'Exp. Cell. Res.', 'Journal of Applied Physics': 'J. Appl. Phys.',\
        'Nature Neuroscience': 'Nat. Neurosci.'
        }

def get_disc_domain():
    disc_domain = {}
    for d, li in domain.items():
        for l in li:
            disc_domain[l] = d
    return disc_domain


def train_j2v(walks, num_features, min_word_count, context, negative = 5):
    num_workers = multiprocessing.cpu_count()
    downsampling = 1e-3
    print("\nTraining model...")
    model = word2vec.Word2Vec(walks, workers=num_workers, size=num_features,min_count = min_word_count, window = context, sample = downsampling, sg = 1, negative = negative)
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "%dfeat_%dminwords_%dcontext" %(num_features, min_word_count, context)
    model.save(model_name)
    print("done and saved model to file!")


def load_j2v(modelfile):
    try:
        model = word2vec.Word2Vec.load(modelfile)
    except:
        import _compat_pickle
        _compat_pickle.IMPORT_MAPPING.update({
            'UserDict': 'collections',
            'UserList': 'collections',
            'UserString': 'collections',
            'whichdb': 'dbm',
            'StringIO':  'io',
            'cStringIO': 'io',
        })
        model = word2vec.Word2Vec.load(modelfile)
    print('\nmodel shape is:', model.syn0.shape, '\n')
    return model


def find_similar(model, vid, vid2vname):
    try:
        journal_score = model.most_similar(vid)
        print("For journal: %s" %(vid2vname[vid]))
        print("The most similar journals or Conferences are:\n")
        for vid, score in journal_score:
            print('{:>80}:\t{:6.5f}'.format(vid2vname[vid], score))
        print()
    except:
        print("opps, \'%s\' is not contained in model! Try to increase # of random walks to cover it!" %(vid2vname[vid]))


def most_close_in_vname(model, keys, by = 'id'):
    vid2vname = mag.get_venue_dict()
    vname2vid = dict((name, vid) for (vid, name) in vid2vname.items())
    if by == 'id':
        for vid in keys:
            if not vid in vid2vname.keys():
                print('opps! found wrong id: %s\n' %(vid))
                continue
            find_similar(model, vid, vid2vname)
    if by == 'name':
        for name in keys:
            try:
                vid = vname2vid[name]
            except:
                print("opps! %s is not in Journal.txt or Conferences.txt at all!\n" %(name))
                continue
            find_similar(model, vid, vid2vname)


def most_close_vector_analogy(model, pos, neg):
    vid2vname = mag.get_venue_dict()
    vname2vid = dict((name, vid) for (vid, name) in vid2vname.items())
    pos_vid = [vname2vid[name] for name in pos]
    neg_vid = [vname2vid[name] for name in neg]
    journal_score = model.most_similar(positive = pos_vid, negative = neg_vid)
    print("For vector analogy: '{}' - '{}' + '{}', possible results are:\n".format(pos[0], neg[0], pos[1]))
    for vid, score in journal_score:
        try: print('{:>80}:\t{:6.5f}'.format(vid2vname[vid], score))
        except: continue


def compute_tsne_2D(model, mapping, perplexity= 30, rand_seed =1, theta= 0.5):
    '''plot the map of science.'''
    vid2index = dict((vid, i) for i, vid in enumerate(model.index2word))
    ix, jids, c_labels = [], [], []
    for i in range(len(mapping)):
        jid, color = mapping.ix[i, ['jid', 'color']]
        if jid in vid2index:
            jids.append(jid)
            ix.append(vid2index[jid])
            c_labels.append(color2hex[color])
    print("Among these venues, %d are covered in the j2v model.\n" %(len(ix)))
    print("Computing t-SNE projection for these journal vectors...")
    X_tsne = tsne(model.syn0[ix, :].astype('float64'), perplexity = perplexity, rand_seed = rand_seed, theta = theta)
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    df = pd.DataFrame(index = range(len(jids)), columns = ['jid'])
    df['jid'] = jids
    df['x'] = X_tsne[:, 0]
    df['y'] = X_tsne[:, 1]
    df['hex_color'] = c_labels
    ffile = 'jid_2d_projection.csv'
    df.to_csv(ffile, sep = ',', header = True, index = False, columns = df.columns, encoding = 'utf-8')
    print('\n2d projection points have written to file: %s' %(ffile))
    # filename = 't-SNE_' + str(perplexity) + '_' +str(theta) + '.pdf'
    # title = "t-SNE 2D projection of {} venue vectors".format(len(X_tsne))
    # util.plot_scatter(X_tsne, c_labels, filename)


def plot_scatter_with_bg_legend(X, c_labels, filename, s = 20, ncol = 4):
    '''2d space scatter plot.
        X has a shape: (n, 2), c_labels is a hex color list.
        This method is used to draw all points and all legends in the same fig.
        Actually, when the legend has too many, it's better to draw them in another fig.
    '''
    hex2index = dict((h, i) for i, h in enumerate(w2v.color2hex.values()))
    index2hex = dict((i, h) for h, i in hex2index.items())
    leng = len(hex2index)
    index = [[] for i in range(leng)]
    bg_dots_index = []
    bg_hex = ''
    for i in range(len(c_labels)):
        col = c_labels[i]
        if col in hex2index.keys():
            index[hex2index[col]].append(i)
        else:
            bg_hex = col
            bg_dots_index.append(i)
    fig = plt.figure(figsize = (8, 6))
    if len(bg_dots_index) > 0: # first plot background points.
        plt.scatter(X[bg_dots_index, 0], X[bg_dots_index, 1], s = 2, color = bg_hex, alpha = 1.0, linewidths = 0)
    sc, labels = [], []
    for i in range(leng):
        if len(index[i]) > 0:
            sc.append(plt.scatter(X[index[i], 0], X[index[i], 1], s = s, color = index2hex[i], alpha = 1.0, linewidths = 0))
            labels.append(w2v.color2dis[w2v.hex2color[index2hex[i]]])
    ld = plt.legend(sc, labels, scatterpoints = 1, loc='best', ncol = ncol, fontsize = 5)
    for i in range(len(sc)):
        ld.legendHandles[i]._sizes = [10]
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.axis('off')
    # plt.title(title)
    fig.savefig(filename, dpi = 300)


def get_direction(model, mapping):
    '''e.g. get direction vector from social science center point to natural science center point.'''
    disc_domain = get_disc_domain()
    vid2index = dict((vid, i) for i, vid in enumerate(model.index2word))
    s_ix, n_ix = [], []
    for i in range(len(mapping)):
        jid,  disc_name = mapping.ix[i, ['jid', 'disc_name']]
        if jid in vid2index and disc_name in disc_domain:
            ind = vid2index[jid]
            if disc_domain[disc_name] == 'natural science':
                n_ix.append(ind)
            else: s_ix.append(ind)
    natural_vec = np.mean(model.syn0[n_ix, :].astype('float64'), axis = 0)
    social_vec = np.mean(model.syn0[s_ix, :].astype('float64'), axis = 0)
    natural_vec = natural_vec / np.linalg.norm(natural_vec)
    social_vec = social_vec / np.linalg.norm(social_vec)
    social2natural = natural_vec - social_vec
    # it doesn't matter normalize the direction or not.
    # social2natural = social2natural / np.linalg.norm(social2natural)
    return social2natural


def cal_cosines(direction, model):
    '''get cosine similarity between each j vector and the direction vector.'''
    cosines = cosine_similarity(model.syn0, direction.reshape(1,-1)).flatten()
    jid_jname = mag.get_journal_dict()
    #most_pos = jid_jname[model.index2word[np.argmax(cosines)]]
    #most_neg = jid_jname[model.index2word[np.argmin(cosines)]]
    #print('most natural journal: %s (%f)' %(most_pos, np.max(cosines)))
    #print('most social journal: %s (%f)' %(most_neg, np.min(cosines)))
    return cosines


def top_ten(cosines, model, type = 'postive direction'):
    '''return top or bottom ten journals based on cosine simi value.'''
    jid_jname = mag.get_journal_dict()
    if type == 'postive direction':
        ten = cosines.argsort()[-11:-1][::-1]
    else: ten = cosines.argsort()[:10]
    for i in ten:
        name = jid_jname[model.index2word[i]]
        score = cosines[i]
        print('{:>80}:\t{:6.5f}'.format(name, score))


def journal_spectrum(name, model, direction):
    jname_jid = mag.get_journal_dict(key = 'name')
    # cosine function returns the (1 - cos) value, which is cosine distance.
    distance = cosine(direction, model[jname_jid[name]])
    print('spectrum value: %f' %(value))
    return 1 - distance


def plot_pid_ref_ax(ax, pid, pid_vid_dict, paper_ref_dict, pid_citation, model):
    '''plot a paper's reference journals in the map of science. all other journal points are light colored as background.'''
    D_value = process_pid(pid, paper_ref_dict, pid_vid_dict, model)
    points = pd.read_csv('jid_2d_projection.csv', header = 0)
    ref_jids = [pid_vid_dict[ref] for ref in paper_ref_dict[pid] if ref in pid_vid_dict]
    ref_jids_set = set(ref_jids)
    count = 0
    for ref in ref_jids_set:
        count += (1 if ref in model else 0)
    index = points['jid'].isin(ref_jids_set)
    points.ix[~index, ['hex_color']] = '#CDC9C9'
    X = (points.ix[:, ['x', 'y']]).as_matrix()
    center = np.array([np.mean((points.ix[index, ['x', 'y']]).as_matrix(), axis=0)])
    X = np.append(X, center, axis=0)
    c_labels = points['hex_color'].tolist()
    c_labels.append('#000000')
    print('%s (%d citations, interdisciplinarity value: %f) has %d references, %d ref has known vids, %d are unique, %d are covered in the j2v model, among which %d have known discipline id (matched in UCSD map)' %(pid, pid_citation[pid], D_value, len(paper_ref_dict[pid]), len(ref_jids), len(ref_jids_set), count, index.sum()))
    util.plot_scatter_with_bg_and_center(ax, X, c_labels)


def plot_pid_ref_map(pid1, pid2, pid_vid_dict, paper_ref_dict, pid_citation, model):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 4.5))
    # fig.subplots_adjust(left = 0.03, right = 0.97, bottom = 0.05, top = 0.95)
    plot_pid_ref_ax(ax1, '7D0EDF01', pid_vid_dict, paper_ref_dict, pid_citation, model) # '802094AE' # high[377]
    plot_pid_ref_ax(ax2, low[22762], pid_vid_dict, paper_ref_dict, pid_citation, model) #7CFC85F9
    plt.show()
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)
    filename = 'two_pid_ref_map.pdf'
    fig.savefig(filename, dpi = 300)


def process_pid(pid, paper_ref_dict, pid_vid_dict, model, metric='euclidean'):
    '''calulate interdisciplinarity (D) value for a pid.'''
    jids = [pid_vid_dict[ref] for ref in paper_ref_dict[pid] if ref in pid_vid_dict]
    j_vecs = np.array([model[jid] for jid in jids if jid in model])
    if j_vecs.shape[0] >= 1:
        return util.mean_euclidean_dis_to_centroid(j_vecs)
    else:
        return False


def process_year_pids(pids, paper_ref_dict, pid_vid_dict, model, metric = 'euclidean'):
    '''calculate D value for all pids.'''
    D_list = dict()
    for pid in pids:
        score = process_pid(pid, paper_ref_dict, pid_vid_dict, model, metric)
        if score:
            if metric == 'median': D_list[pid] = np.median(score)
            elif metric == 'euclidean': D_list[pid] = score
    if len(D_list) < 1: sys.exit('this group is empty!')
    return D_list


def year_interdisciplinarity(D_list, pid_citation, qua1 = 50, qua2 = 99.9):
    '''group pids into three groups based on # of citations'''
    year_citation = [pid_citation[pid] for pid in D_list]
    first, second = np.percentile(year_citation, [qua1, qua2])
    high, middle, low = [], [], []
    for pid in D_list.keys():
        if pid_citation[pid] <= first:
            low.append(pid)
        elif pid_citation[pid] <= second:
            middle.append(pid)
        else: high.append(pid)
    high_D = np.array([D_list[pid] for pid in high])
    middle_D = np.array([D_list[pid] for pid in middle])
    low_D = np.array([D_list[pid] for pid in low])
    print('high citation group: {}, middle citation group: {}, low citation group: {}'.format(len(high), len(middle), len(low)))
    plot_three_CDF([high_D.copy(), middle_D.copy(), low_D.copy()], first, second)
    return high, middle, low, high_D, middle_D, low_D


def plot_three_CDF(groups, first, second):
    '''cumulative distribution plot of three list of numbers'''
    fig, ax = plt.subplots(figsize = (6, 4))
    fig.subplots_adjust(left = 0.15, bottom = 0.15)
    colors = ['r', 'g', 'b']
    labels = []
    labels.append('C >= %d' %(second))
    labels.append('%d <= C < %d' %(first, second))
    labels.append('C < %d' %(first))
    for g, c, l in zip(groups, colors, labels):
        g.sort()
        g = np.concatenate((g, g[[-1]]))
        plt.step(g, np.arange(1, g.size+1)/float(g.size), color=c, label=l)
    legend = plt.legend(loc='best', labelspacing = 0.2, shadow=False, frameon = False)
    for label in legend.get_texts():
        label.set_fontsize(8)
    for label in legend.get_lines():
        label.set_linewidth(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.tick_params(axis='x', which = 'both',length=0)
    # xticks = ax.xaxis.get_major_ticks()
    # xticks[-1].label1.set_visible(False)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #ax.arrow(0.8, -0.05, 0.08, 0, width=0.000001, color="k", clip_on=False, head_width=0.01, head_length=0.06)
    #ax.arrow(0, 0.95, 0, 0.08, width=0.000001, color="k", clip_on=False, head_width=0.01, head_length=0.08)
    ax.set_xlabel(r'$I$')
    ax.set_ylabel(r'$P \; (\leq I)$')
    # ax.text(-0.1, 1.18, letter, ha='left', va='top', transform=ax.transAxes)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    fn ='h%d_m%d_l%d_cdf.pdf' %(len(groups[0]), len(groups[1]), len(groups[2]))
    fig.savefig(fn, dpi = 300)


def four_groups_jid_pairs_dis(model, mapping, num_pairs = 100000):
    '''generate journal pair cosine similarity.'''
    mapping = mapping[mapping.apply(lambda x: x['jid'] in model.index2word, axis=1)]
    # mapping = mapping.sample(frac=1, random_state= 10)
    grouped = mapping.groupby('disc_name')
    fine_grouped = mapping.groupby('subd_name')
    # within-discipline
    pairs = []
    for name, group in grouped:
        p = util.sample_pairs(group['jid'].tolist(), 8000)
        pairs.extend(p)
    pairs = random.sample(pairs, num_pairs)
    # within-subdiscipline
    pairs_subd = []
    for name, subd_group in fine_grouped:
        p = list(combinations(subd_group['jid'].tolist(), 2))
        pairs_subd.extend(p)
    pairs_subd = random.sample(pairs_subd, num_pairs)
    # cross-subdiscipline
    sub_groups = []
    for name, group in fine_grouped:
        sub_groups.append(group['jid'].tolist())
    sub_accross_pairs = util.sample_cross_group_pairs(sub_groups, 20)
    sub_accross_pairs = random.sample(sub_accross_pairs, num_pairs)
    # random pairs
    random_p = util.sample_pairs(mapping['jid'].tolist(), num_pairs)
    # calculate cosine similarity
    distances = [model.similarity(p1, p2) for (p1, p2) in pairs]
    subd_distances = [model.similarity(p1, p2) for (p1, p2) in pairs_subd]
    sub_cross_distances = [model.similarity(p1, p2) for (p1, p2) in sub_accross_pairs]
    rand_distances = [model.similarity(p1, p2) for (p1, p2) in random_p]
    return distances, subd_distances, sub_cross_distances, rand_distances


def plot_groups_similarity(distances, subd_distances, sub_cross_distances, rand_distances):
    '''plot histgram of these pairs' cosine similarity.'''
    f, (ax1, ax2)= plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8, 5))
    f.subplots_adjust(left = 0.15, bottom = 0.1)
    ax1.hist(distances, bins = 50, alpha = 0.5, label = 'within-discipline',  color = '#1b9e77')
    ax1.hist(rand_distances, bins = 50, alpha = 0.5, label = 'random', color = '#d95f02')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('cosine similarity')
    ax1.legend(loc='best', fontsize = 8, frameon = False)
    # ax1.set_title('within-discipline journal pairs')
    ax2.hist(subd_distances, bins = 50, alpha = 0.5, label = 'within-sub', color = '#1b9e77')
    ax2.hist(rand_distances, bins = 50, alpha = 0.5, label = 'random', color = '#d95f02')
    # ax2.set_title('within-subdiscipline journal pairs')
    ax2.set_xlabel('cosine similarity')
    ax2.legend(loc='best', fontsize = 8, frameon = False)
    plt.show()
    f.savefig('similarity.pdf', dpi = 300)
