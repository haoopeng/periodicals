""" code for playing with MS Academic Graph """
#!/usr/bin/env python
# encoding: utf-8
# author: Hao Peng
# email: pengh1992@gmail.com

import os
import math
import csv
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import j2v


DATA_ROOT = '/l/nx/data/haopeng/j2v/'
#DATA_ROOT = '/Users/apple/Desktop/journal2vec/data/'
MAG_Ref_FILE = 'PaperReferences.txt'
MAG_Papers_FILE = 'Papers.txt'
MAG_Journals_FILE = 'Journals.txt'
MAG_Conferences_FILE = 'Conferences.txt'
UCSD_MAP_TABLE = 'UCSDmapDataTables.xlsx'


def get_path(filename):
    '''return absolute file path'''
    return os.path.join(DATA_ROOT, filename)

def pow10ceil(x):
    # 101, 500 -> 1000
    return 10**math.ceil(math.log(x, 10))

def yield_one_line(filename, delimiter = '\t'):
    '''a generator which produce one line of a given file'''
    filepath = get_path(filename)
    with open(filepath, 'r') as file:
        print('processing %s...' %(filepath))
        reader = csv.reader(file, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        count = 0
        for row in reader:
            count += 1
            if count % 1000000 == 0:
                print('processed %d lines...' %(count))
            yield row
        print('finished processing!\n')

def filter_refs(pfile, rfile, ofile, pdelimiter = ',', rdelimiter = ','):
    """ extract references between the papers in the paper_set """
    pid2vid, _ = get_pid_vid_dict(pfile, pdelimiter)
    paper_set = set(list(pid2vid.keys()))
    del(pid2vid)
    count = 0
    with open(get_path(ofile), 'w') as out:
        for row in yield_one_line(rfile, rdelimiter):
            spid, tpid = row
            if (spid in paper_set) and (tpid in paper_set):
                count += 1
                out.write(rdelimiter.join(row) + '\n')
    print('get %s citations!' %(count))


# def get_citation(pfile, rfile, pdelimiter = ',', rdelimiter = ',', by = 'pid'):
#     '''return a journal citation counts dictionary. with jid as keys and all its papers' citation counts as values'''
#     pid_citation = defaultdict(int)
#     for row in yield_one_line(rfile, rdelimiter):
#         spid, tpid = row
#         pid_citation[tpid] += 1
#     if by == 'pid':
#         print('got paper citation count!')
#         return pid_citation
#     elif by == 'jid':
#         jid_count = defaultdict(list)
#         _, pid_vid_dict = get_set_dict(pfile, pdelimiter)
#         for pid in pid_vid_dict.keys():
#             jid = pid_vid_dict[pid]
#             jid_count[jid].append(pid_citation[pid])
#         print('got journal citation count!')
#         return jid_count


def get_paper_ref_dict_and_citation(rfile, delimiter = ','):
    paper_ref_dict = defaultdict(list)
    pid_citation = defaultdict(int)
    count = 0
    for row in yield_one_line(rfile, delimiter):
        count += 1
        spid, tpid = row
        pid_citation[tpid] += 1
        paper_ref_dict[spid].append(tpid)
    print("pid refecence dict and citation count is prepared! there are %s citations in total." %(count))
    return paper_ref_dict, pid_citation


def get_pid_vid_dict(pfile, delimiter = ','):
    '''get a paper pid set and a pid to jid dict from filtered paper file'''
    pid_vid_dict = dict()
    year_pids = defaultdict(list)
    for row in yield_one_line(pfile, delimiter):
        pid, year, jid, cid = row
        if jid != '':
        # jid should not be '' when train a j2v model, thus please filter paper and references first. But jid could be '' when measuring interdisciplinarity, because we have to process the whole data.
            pid_vid_dict[pid] = jid
        else: pid_vid_dict[pid] = cid
        year_pids[year].append(pid)
    print('pid2vid dict and year-pid dict are prepared now! It has %d papers!\n' %(len(pid_vid_dict)))
    return pid_vid_dict, year_pids


def get_journal_dict(key = 'id', verbose = False):
    '''return a mapping dictionary from jid to jname or vice versa(with jname as keys and jid as values)'''
    jid2jname = {}
    li = set()
    for row in yield_one_line(MAG_Journals_FILE):
        jid, jname = row
        if jname in li:
            continue
        li.add(jname)
        jid2jname[jid] = jname
    if verbose:
        print('journal dict with size of %d is loaded!\n' %(len(jid2jname)))
    jname2jid = dict((jname, jid) for (jid, jname) in jid2jname.items())
    if key == 'id':
        return jid2jname
    if key == 'name':
        return jname2jid


def get_conference_dict(key = 'id', verbose = False):
    '''return a mapping dictionary from jid to jname or vice versa(with jname as keys and jid as values)'''
    cname2cid = {}
    li = set()
    for row in yield_one_line(MAG_Conferences_FILE):
        cid, short, cname = row
        if cname in li:
            try:
                del cname2cid[cname]
                continue
            except: continue
        li.add(cname)
        cname2cid[cname] = cid
    if verbose:
        print('Conference dict with size of %d is loaded!\n' %(len(cname2cid)))
    cid2cname = dict((cid, name) for (name, cid) in cname2cid.items())
    if key == 'id':
        return cid2cname
    if key == 'name':
        return cname2cid

def get_venue_dict(key = 'id'):
    jdict = get_journal_dict(key)
    cdict = get_conference_dict(key)
    vdict = dict(list(jdict.items()) + list(cdict.items()))
    # print('got venue dict! size %d\n' %(len(vdict)))
    return vdict


def reduce_paper_file(pfile, out_file, pdelimiter = '\t', odelimiter = ','):
    ''' it basiclly keep only four fields of the original paper file. Thus reduce the file size from 29GB to 2GB.
    '''
    with open(get_path(out_file), 'w') as out:
        for row in yield_one_line(pfile, pdelimiter):
            pid = row[0]
            year = row[3]
            jid = row[8]
            vid = row[9]
            paper = [pid, year, jid, vid]
            out.write(odelimiter.join(paper) + '\n')

def clean_reference(rfile, ofile, rdelimiter = '\t', odelimiter = ','):
    ''' remove cycle citations in reference file'''
    first_set = set()
    total = 0
    for row in yield_one_line(rfile, rdelimiter):
        total += 1
        first_set.add(odelimiter.join(row))
    with open(get_path(ofile), 'w') as out:
        count = 0
        for row in yield_one_line(rfile, rdelimiter):
            spid, tpid = row
            if not tpid+odelimiter+spid in first_set:
                line = odelimiter.join(row) + '\n'
                count += 1
                out.write(line)
    print('remomved %d cycles, saved %d citations!' %(total-count, count))


def filter_paper_year_jid(pfile, ofile, year, delimiter=',', jids='all'):
    '''
        * jids should be a set, other than list, which runs faster
        * year is like [1950, 2000]
    '''
    if jids == 'all':
        jids = set(get_journal_dict().keys())
    start, end = year
    count = 0
    with open(get_path(ofile), 'w') as out:
        for row in yield_one_line(pfile, delimiter):
            # pid = row[0]
            y = int(row[1])
            jid = row[2]
            cid = row[3]
            if jid !='' and cid =='' and y >= start and y <= end and (jid in jids):
                count += 1
                out.write(delimiter.join(row) + '\n')
    print('get %s papers!' %(count))


def filter_paper(pfile, ofile, delimiter = ','):
    count = 0
    with open(get_path(ofile), 'w') as out:
        for row in yield_one_line(pfile, delimiter):
            # pid = row[0]
            jid = row[2]
            cid = row[3]
            if (jid !='' and cid == '') or (cid != '' and jid == ''):
                count += 1
                out.write(delimiter.join(row) + '\n')
    print('get %s papers!' %(count))


def get_venue_year(pfile, delimiter = ','):
    year_vids = defaultdict(set)
    vids_year = defaultdict(lambda: 2017)
    for row in yield_one_line(pfile, delimiter):
        pid = row[0]
        year = int(row[1])
        jid = row[2]
        cid = row[3]
        vid = jid if jid != '' else cid
        if year < vids_year[vid]:
            vids_year[vid] = year
    for vid, year in vids_year.items():
        year_vids[year].add(vid)
    print('got year vid dict!')
    return year_vids


def get_paper_name(pids):
    pid_pname = dict()
    pids = set(pids)
    for row in mag.yield_one_line('Papers.txt', delimiter='\t'):
        pid = row[0]
        name = row[1]
        if pid in pids:
            pid_pname[pid] = name
    return pid_pname


def get_num_refs(pids, pid_vid_dict, paper_ref_dict):
    num_refs = defaultdict(int)
    for pid in pids:
        jids = [pid_vid_dict[ref] for ref in paper_ref_dict[pid] if ref in pid_vid_dict]
        for jid in jids:
            num_refs[pid] += (1 if jid in model else 0)
    return num_refs


def get_num_authors(pids):
    Paper_author_file = 'PaperAuthorAffiliations.txt'
    num_authors = defaultdict(int)
    pids = set(pids)
    for row in mag.yield_one_line(filename = Paper_author_file):
        pid = row[0]
        if pid in pids:
            num_authors[pid] += 1
    return num_authors


def get_num_citations(pids, pid_citation):
    num_citations = dict((pid, pid_citation[pid]) for pid in pids)
    return num_citations


def random_walks(num_walks, max_length, pid_vid_dict, paper_ref_dict, pids = 'all'):
    '''generate a specific number of random walks, as the raw reference file has triangle citation even after cleaning cycle citation, we just terminate and discard a walk once a pid recurred in the walk during the walking. We also discard walks with length one, as they provide no infromation about the journal citation flow.'''
    walks = []
    if pids == 'all':
        pids = list(paper_ref_dict.keys())
    s = 0
    print('random walking now...')
    while s < num_walks:
        pwalk = []
        jwalk = []
        pid = random.choice(pids)
        jid = pid_vid_dict[pid]
        pwalk.append(pid)
        jwalk.append(jid)
        i = 0
        while i < max_length:
            if len(paper_ref_dict[pid]) == 0:
                break
            pid = random.choice(paper_ref_dict[pid])
            jid = pid_vid_dict[pid]
            if pid in pwalk:
                jwalk = []
                break
            pwalk.append(pid)
            jwalk.append(jid)
            i += 1
        if len(jwalk) > 1:
            s += 1
            if s % 1000000 == 0:
                print("{}/{} generated...".format(s, num_walks))
            walks.append(jwalk)
    return walks


def walk_len_freq(walks):
    c = Counter([len(walk) for walk in walks])
    fig = plt.figure(figsize = (6, 4))
    fig.subplots_adjust(left = 0.15, bottom = 0.15)
    # plt.hist(li, bins = 50, normed = 0, facecolor = 'b', alpha = 0.5)
    ax = plt.gca()
    x = [float(k) for k in c.keys()]
    y = [float(v) for v in c.values()]
    ax.scatter(x, y, c = 'blue', alpha = 0.5) # edgecolor = 'g'
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.8, pow10ceil(max(x))])
    ax.set_ylim([1e-1, pow10ceil(max(y))])
    ax.set_xlabel('walk length')
    ax.set_ylabel('frequency')
    # ax.set_title("%d random walk length distribution" %(len(walks)))
    plt.grid(True)
    fig.savefig('walk-length-dist.pdf', dpi = 300)
    print("\n%d random walks' length freq fig saved to file!\n" %(len(walks)))


def get_journal_discipline_map(ucsd_file, out_file, delimiter = ','):
    ucsd_file = get_path(ucsd_file)
    journal2dis = pd.read_excel(ucsd_file, sheetname = 'Table 4', header = 12, index_col = None, parse_cols = 3)
    subid2id = pd.read_excel(ucsd_file, sheetname = 'Table 1', header = 12, index_col = None, parse_cols = 2)
    did2dname = pd.read_excel(ucsd_file, sheetname='Table 2', header= 12, index_col = None, parse_cols = [0, 1, 4])
    jname2jid = get_journal_dict(key='name')
    li = list()
    total = len(jname2jid)
    count = 0
    for jname in jname2jid.keys():
        count += 1
        if count % 100 == 0:
            print('processed {}/{}...'.format(count, total))
        for i, row in journal2dis.iterrows():
            ucsd_jname = row['formal_name']
            if fuzz.ratio(ucsd_jname, jname) >= 90:
                # print(jname, '||', ucsd_jname)
                li.append((jname2jid[jname], jname, ucsd_jname,row['subd_id']))
                break
    li = pd.DataFrame(li, columns=['jid', 'jname', 'ucsd_jname', 'subd_id'])
    result = pd.merge(li, subid2id, how='inner', on = ['subd_id'])
    result = pd.merge(result, did2dname, how='inner', on = ['disc_id'])
    matched = result.ucsd_jname.tolist()
    visited = set()
    for i, row in journal2dis.iterrows():
        ucsd_jname = row['formal_name']
        jfraction = float(row['jfraction'])
        if ucsd_jname in matched and jfraction != 1.0 and ucsd_jname not in visited:
            result.loc[result['ucsd_jname'] == ucsd_jname, ['subd_id', 'subd_name', 'disc_id', 'disc_name', 'color']] = [0, 'Interdiscipline', 0, 'Interdiscipline', 'Black']
            visited.add(ucsd_jname)
    result.to_csv(get_path(out_file), sep = delimiter, header = True, index = False, columns = result.columns, encoding = 'utf-8')

    print('journal to discipline map saved to file!')

def load_map_jid_discipline(map_file = 'journal_discipline_map.csv'):
    mapping = pd.read_csv(get_path(map_file), header = 0)
    mapping = mapping.drop_duplicates(subset = ['jid'])
    mapping.index = range(len(mapping))
    num_inter = len(mapping[mapping['disc_name'] == 'Interdiscipline'])
    print("%d journals in MAG's Journal.txt were matched to UCSD data, and %d of them are interdisciplinary journals in UCSD catelog." %(len(mapping), num_inter))
    return mapping

