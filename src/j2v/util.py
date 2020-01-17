#! /usr/bin/env python
# coding: utf-8

import random
import math

def comb(k):
    '''compute row and column of a k-th element in a lower-triangle'''
    row = int((math.sqrt(1+8*k)+1)/2)
    column = int(k-(row-1)*(row)/2)
    return [row,column]


def mean_euclidean_dis_to_centroid(a):
    '''compute square root euclidean dis to centroid vectors.'''
    return np.mean(np.sqrt(np.sum(np.square(a-np.mean(a, axis=0)), axis = 1)))


def sample_pairs(li, num = 1000):
    '''if len(li) is small, we can not sample equal num of pairs from li. So it is better to generate all pairs, and then combine all them together to select a specific # of pairs from the combined one; if len(li) is large, we should sample equal num in each li'''
    total = int(len(li) * (len(li) - 1) / 2)
    if total > num:
        index = random.sample(range(total), num)
    else:
        print('This group is small, can not generate # of pairs required, return all pairs!')
        index = range(total)
    pairs = []
    for i in index:
        row, column = comb(i)
        pairs.append((li[row], li[column]))
    return pairs


def sample_cross_group_pairs(list_of_groups, num_pairs):
    num_groups = len(list_of_groups)
    cross_pairs = []
    for i in range(num_groups):
        for j in range(i+1, num_groups):
            len1, len2 = len(list_of_groups[i]), len(list_of_groups[j])
            total =  len1 * len2
            if num_pairs > total:
                index = range(total)
            else:
                index = random.sample(range(total), num_pairs)
            for k in index:
                row = int(k/len2)
                col = k % len2
                cross_pairs.append((list_of_groups[i][row], list_of_groups[j][col]))
    return cross_pairs


def plot_scatter(X, c_labels, filename, s = 3):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.scatter(X[:, 0], X[:, 1], s = s, color = c_labels, alpha = 1.0, linewidths = 0)
    ax.set_axis_off()
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    # hide specific tick label on a ax.
    # xticks = ax.xaxis.get_major_ticks()
    # xticks[-1].label1.set_visible(False)
    fig.savefig(filename, dpi = 300)


def plot_scatter_with_bg_and_center(ax, X, c_labels):
    '''here, we plot for a ax, not a fig! You can change as you want!'''
    bg_index, fg_index = [], []
    bg_c, fg_c = [], []
    bg_c = '#CDC9C9'
    center_c = '#000000'
    center_index = 0
    for i in range(len(c_labels)):
        hex_c = c_labels[i]
        if hex_c == center_c:
            center_index = i
        elif hex_c == bg_c:
            bg_index.append(i)
        else:
            fg_index.append(i)
            fg_c.append(hex_c)
    ax.scatter(X[bg_index, 0], X[bg_index, 1], s = 2, color = bg_c, alpha = 1.0, linewidths = 0)
    ax.scatter(X[fg_index, 0], X[fg_index, 1], s = 20, color = fg_c, alpha = 1.0, linewidths = 0)
    ax.scatter(X[center_index, 0], X[center_index, 1], s = 20, color = center_c, marker='x')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def circle(xy, radius, kwargs=None):
    from matplotlib import patches
    import matplotlib.pyplot as plt
    """Create circle on figure with axes of different sizes.

    Plots a circle on the current axes using `plt.Circle`, taking into account
    the figure size and the axes units.

    It is done by plotting in the figure coordinate system, taking the aspect
    ratio into account. In this way, the data dimensions do not matter.
    However, if you adjust `xlim` or `ylim` after plotting `circle`, it will
    screw them up; set `plt.axis` before calling `circle`.

    Parameters
    ----------
    xy, radius, kwars :
        As required for `plt.Circle`.
        Note: radius is ratio, not abs value.

    """

    # Get current figure and axis
    fig = plt.gcf()
    ax = fig.gca()

    # Calculate figure dimension ratio width/height
    pr = fig.get_figwidth()/fig.get_figheight()

    # Get the transScale (important if one of the axis is in log-scale)
    tscale = ax.transScale + (ax.transLimits + ax.transAxes)
    ctscale = tscale.transform_point(xy)
    cfig = fig.transFigure.inverted().transform(ctscale)

    # Create circle
    if kwargs == None:
        circ = patches.Ellipse(cfig, radius, radius*pr,
                transform=fig.transFigure)
    else:
        circ = patches.Ellipse(cfig, radius, radius*pr,
                transform=fig.transFigure, **kwargs)

    # Draw circle
    ax.add_artist(circ)
