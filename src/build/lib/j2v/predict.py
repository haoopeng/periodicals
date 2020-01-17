#! /usr/bin/env python
# coding: utf-8

from j2v import mag
from j2v import w2v
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def prepare_data(model, mapping):
    '''model keys and mapping keys only overlap some vids.'''
    jid2index = dict((jid, i) for i, jid in enumerate(model.index2word))
    data, label, dis_id, new_data = dict(), dict(), dict(), dict()
    for i in range(len(mapping)):
        jid, disc_name, did = mapping.ix[i, ['jid', 'disc_name', 'disc_id']]
        if jid in model.index2word:
            data[jid] = model.syn0[jid2index[jid], :]
            label[jid] = disc_name
            dis_id[jid] = did
    for jid in model.index2word:
        if jid not in data.keys():
            new_data[jid] = model.syn0[jid2index[jid], :]
    data = pd.DataFrame(list(data.items()), columns = ['jid', 'vec'])
    label = pd.DataFrame(list(label.items()), columns = ['jid', 'disc_name'])
    dis_id = pd.DataFrame(list(dis_id.items()), columns = ['jid', 'disc_id'])
    data = pd.merge(data, label, on = ['jid'], how = 'inner')
    data = pd.merge(data, dis_id, on = ['jid'], how = 'inner')
    new_data = pd.DataFrame(list(new_data.items()), columns = ['jid', 'vec'])
    print('data length: %d, new_data length: %d' %(len(data), len(new_data)))
    return data, new_data


def base_line(data):
    length = len(data)
    grouped = data.groupby('disc_name')
    print('\t\t{} known data distribution (baseline prediction)\n'.format(length))
    for name, group in grouped:
        le = group.shape[0]
        print('{:>45}:\t{:4} ({:>5.2f}/100)'.format(name, le, le/length*100))


def split_data(data):
    data = data.sample(frac = 1.0)
    first = int(len(data) * 0.8)
    second = int(len(data) * 0.9)
    train = data[:first]
    dev = data[first:second]
    test = data[second:]
    train.index = np.arange(len(train))
    dev.index = np.arange(len(dev))
    test.index = np.arange(len(test))
    print("\ndata length - (train: %d, dev: %d, test: %d)" %(len(train), len(dev), len(test)))
    return train, dev, test

def get_feat_label(data):
    data_feat = np.array(data['vec'].tolist())
    y_true = data['disc_name']
    return data_feat, y_true

def evaluate(algo, data):
    data_feat, y_true = get_feat_label(data)
    y_pred = algo.predict(data_feat)
    y_pred = y_pred.tolist()
    y_true = y_true.tolist()
    labels = list(set(y_true))
    confux = metrics.confusion_matrix(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred, target_names=labels)
    print('\nConfusion matrix:\n\n{}'.format(confux))
    print('\nReport:\n\n{}'.format(report))


def predict_to_file(algo, new_data, out_file, delimiter = ','):
    new_data_feat = np.array(new_data['vec'].tolist())
    pred_disc_name = algo.predict(new_data_feat)
    vid2vname = mag.get_venue_dict()
    new_data['venue_name']=[vid2vname[jid] for jid in new_data['jid'].tolist()]
    new_data['disc_name'] = pred_disc_name
    new_data['color'] = [w2v.dis2color[dis] for dis in pred_disc_name]
    new_data.to_csv(out_file, sep = delimiter, header = True, index = False, columns = ['jid', 'venue_name', 'disc_name', 'color'], encoding = 'utf-8')
    print('prediction written to file!')

def algorithm(model = 'knn', n_neighbors = 10, n_estimators = 10, C = 10):
    if model == 'knn':
        print('training knn model...')
        algo = KNeighborsClassifier(n_neighbors = n_neighbors)
    elif model == 'lr':
        print('training logistic regression model...')
        algo = LogisticRegression('l2', C = C)
    elif model == 'rf':
        print('training random forest model...')
        algo = RandomForestClassifier(n_estimators = n_estimators)
    return algo


def load_pred_discipline(pred_file, delimiter = ','):
    prediction = pd.read_csv(pred_file, sep = delimiter, header = 0, encoding = 'utf-8')
    return prediction

def get_combined_mapping(model, mapping, prediction):
    name2id = mapping.ix[:, ['disc_name', 'disc_id']].set_index('disc_name')['disc_id'].to_dict()
    prediction['disc_id'] = [name2id[name] for name in prediction['disc_name'].tolist()]
    combined_mapping = pd.concat([mapping, prediction], axis = 0)
    combined_mapping.index = range(len(combined_mapping))
    # combined_mapping is larger than length of model. :)
    return combined_mapping

