# Author: Aurélien Renault <aurelien.renault@outlook.fr>
# inspired from https://github.com/hfawaz/cd-diagram
# License: GPL3
import os
import numpy as np
import pandas as pd
import matplotlib
import warnings
#from pandas.core.common import SettingWithCopyWarning

#matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

# from utils import check_df_format

def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None, highlight="",
                width=6, textspace=1, reverse=False, filename=None, labels=False, scores=None, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a
    
    if cd is not None:
        def get_lines(sums, hsd):
            # get all pairs
            lsums = len(sums)
            allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
            # remove not significant
            notSig = [(i, j) for i, j in allpairs
                      if abs(sums[i] - sums[j]) <= hsd]
            # keep only longest
            def no_longer(ij_tuple, notSig):
                i, j = ij_tuple
                for i1, j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

            return longest

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 2) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=1)

    bigtick = 0.2
    smalltick = 0.1
    linewidth = 1.0
    linewidth_sign = 3.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=1)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=12)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24
    
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        list_up = [x.upper() for x in highlight] if isinstance(highlight, list) else [highlight.upper()]
        list_low = [x.lower() for x in highlight] if isinstance(highlight, list) else [highlight.lower()]
        list_va = [x for x in highlight] if isinstance(highlight, list) else [highlight]
        
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels=='rank':
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="right", va="center", size=12)
        elif labels=='metric':
            text(textspace + 0.3, chei - 0.075, format(scores[i]*100, '.2f'), ha="right", va="center", size=12)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=15)
        
        if filter_names(nnames[i]) in list_up + list_low + list_va:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace - 0.1, chei)],
                 linewidth=linewidth, color='red')
            if labels=='rank':
                text(textspace + 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="right", va="center", size=12, c='red')
            elif labels=='metric':
                text(textspace + 0.3, chei - 0.075, format(scores[i]*100, '.2f'), ha="right", va="center", size=12, c='red')
            text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=15, c='red')

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels=='rank':
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="left", va="center", size=12)
        elif labels=='metric':
            text(textspace + scalewidth - 0.3, chei - 0.075, format(scores[i]*100, '.2f'), ha="left", va="center", size=12)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),ha="left", va="center", size=15)
        
        if filter_names(nnames[i]) in list_up + list_low +list_va:
            line([(rankpos(ssums[i]), cline),
                  (rankpos(ssums[i]), chei),
                  (textspace + scalewidth + 0.1, chei)],
                 linewidth=linewidth, color='red')
            if labels=='rank':
                text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="left", va="center", size=12, c='red')
            elif labels=='metric':
                text(textspace + scalewidth - 0.3, chei - 0.075, format(scores[i]*100, '.2f'), ha="left", va="center", size=12, c='red')
            text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]), ha="left", va="center", size=15, c='red')
    
    if cd is None:
        # no-significance lines
        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2

            for l, r in lines:
                line([(rankpos(ssums[l]) - side, start),
                      (rankpos(ssums[r]) + side, start)],
                     linewidth=linewidth_sign)
                start += height
                print('drawing: ', l, r)

        # draw_lines(lines)
        start = cline + 0.2
        side = -0.02
        height = 0.1

        # draw no significant lines
        # get the cliques
        cliques = form_cliques(p_values, nnames)
        i = 1
        achieved_half = False
        #print(nnames)
        for clq in cliques:
            if len(clq) == 1:
                continue
            #print(clq)
            min_idx = np.array(clq).min()
            max_idx = np.array(clq).max()
            if min_idx >= len(nnames) / 2 and achieved_half == False:
                start = cline + 0.25
                achieved_half = True
            line([(rankpos(ssums[min_idx]) - side, start),
                  (rankpos(ssums[max_idx]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
    else:
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        #line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        #line([(begin, distanceh + bigtick / 2),
        #      (begin, distanceh - bigtick / 2)],
        #     linewidth=linewidth_sign)
        #line([(end, distanceh + bigtick / 2),
        #      (end, distanceh - bigtick / 2)],
        #     linewidth=linewidth_sign)
        #text((begin + end) / 2, distanceh - 0.05, "CD",
        #     ha="center", va="bottom")

        # no-significance lines
        def draw_lines_cd(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l, r in lines:
                line([(rankpos(ssums[l]) - side, start),
                      (rankpos(ssums[r]) + side, start)],
                     linewidth=2.5)
                start += height 
        draw_lines_cd(lines)


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagram(df, alpha=0.05, labels=False, save=None,
                    title=False, style=None, verbose=True, highlight="", mode='wilco_holm', 
                    pairwise_matrix=None, ascending=False, full=False):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    
    df_perf : pd.Dataframe containing hypothesis one want to compare (datasets col should be named 'Name', see line 565)
    metric : str, based on which metric to compare your hypothesis (should be in df_perf columns)
    obj_ranked : str, the objects to compare, appearing on the diagram (should be in df_perf columns)
    alpha : float, precision of statistical tests (nemenyi only supports 0.05 and 0.1)
    labels : one of ['rank', 'metric', False] => label cd with mean rank / metric or with nothing
    path : str, if provided save figure on path
    title : str, whether to add some title
    style : one of ['upper', 'lower', None] displayed name to be uppercase / lowercase or as provided 
            (change df_perf !)
    verbose : if True display exact p-values in console
    highlight : str/list of str Object one want to highlight in red
    mode : ['nemenyi', 'wilco_holm', None] 
    pairwise_matrix : ['corrected', 'uncorrected', None] parwise comparison with or without Holm's correction
    ascending : if True, the lower the better ; if False the higher the better
    full : if True display full ranks (from 1 to n_hypothesis) else adapt to (min(ranks), max(ranks))
    
    """
    
    # check_df_format(df) ? no such function in utils 
    
    #warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    if style=='upper':
        df['competitor'] = df['competitor'].str.upper()
    elif style=='lower':
        df['competitor'] = df['competitor'].str.lower()
    elif style=='title':
        df['competitor'] = df['competitor'].str.title()

    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df, alpha=alpha, verbose=verbose, asc=ascending)
    
    if verbose:
        print(average_ranks)

        for p in p_values:
            print(p)
            
    if pairwise_matrix is not None:
        
        average_ranks_tmp = average_ranks.iloc[::-1]
        pairewise_comparison_df = pd.DataFrame(
            np.zeros(shape=(len(average_ranks_tmp.index),len(average_ranks_tmp.index))),
            columns=average_ranks_tmp.index,
            index=average_ranks_tmp.index
        )
        if pairwise_matrix=='corrected':
            for p in p_values:
                if p[3]:
                    pairewise_comparison_df[p[0]][p[1]] = 1
                    pairewise_comparison_df[p[1]][p[0]] = 1
        elif pairwise_matrix=='uncorrected':
            for p in p_values:
                if p[2] < alpha:
                    pairewise_comparison_df[p[0]][p[1]] = 1
                    pairewise_comparison_df[p[1]][p[0]] = 1
    
        plt.imshow(pairewise_comparison_df, cmap=plt.cm.gray)
        plt.xticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, rotation='vertical', fontsize=15)
        plt.yticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, fontsize=15)
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        if save:
            if pairwise_matrix is not None:
                if not os.path.isdir(os.path.join(save, 'pairwise_matrix')):
                    os.mkdir(os.path.join(save, 'pairwise_matrix'))   
                path_mat = os.path.join(save, 'pairwise_matrix')
                plt.savefig(os.path.join(path_mat, f'pair-matrix-{pairwise_matrix}.pdf'), bbox_inches='tight', dpi=250)
        plt.show()
        plt.close()
    
    if labels=='metric':
        scores = []
        for obj in average_ranks.keys():
            if ascending:
                scores.append(df[df['competitor']==obj]['metric'].mean() * 1e-2) 
            else:
                scores.append(df[df['competitor']==obj]['metric'].mean())       
    else:
        scores=None
    
    if full:
        lowv = None
        highv = None
    else:
        lowv = max(int(math.floor(min(average_ranks.values))) - 1, 1) if round(min(average_ranks.values))==math.floor(min(average_ranks.values)) else int(math.floor(min(average_ranks.values)))
        highv = min(int(math.ceil(max(average_ranks.values))) + 1, len(average_ranks)) if round(max(average_ranks.values))==math.ceil(max(average_ranks.values)) else int(math.ceil(max(average_ranks.values)))
    
    if mode=='wilco_holm':
        graph_ranks(average_ranks.values, average_ranks.keys(), p_values, lowv=lowv, highv=highv, highlight=highlight,
                    cd=None, reverse=True, width=8, textspace=2, labels=labels, scores=scores)
    elif mode=='nemenyi':
        n = df.groupby(['competitor']).size()[0]
        cd = compute_CD(average_ranks, n, alpha=str(alpha))
        graph_ranks(average_ranks.values, average_ranks.keys(), None, lowv=lowv, highv=highv, highlight=highlight,
                    cd=cd, reverse=True, width=8, textspace=2, labels=labels, scores=scores)
    elif mode==None:
        pass
    else:
        raise ValueError("mode parameter should be either nemenyi or wilco_holm")

    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
    if title:
        plt.title(title, fontdict=font, y=0.9, x=0.5)
    if save:
        if mode is not None:
            plt.savefig(os.path.join(save, f'cd-diagram-{mode}.pdf'), bbox_inches='tight', dpi=250)
    plt.show()
    plt.close()
    
def wilcoxon_holm(alpha=0.05, df_perf=None, verbose=True, asc=False):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    if verbose:
        print(pd.unique(df_perf['competitor']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['competitor']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['competitor'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['competitor'] == c]['metric'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
        #exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['competitor'] == classifier_1]['metric']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['competitor'] == classifier_2]
                              ['metric'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='zsplit')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['competitor'].isin(classifiers)].sort_values(['competitor', 'dataset'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['metric']).reshape(m, max_nb_datasets)
    
    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers))#, columns=
    #np.unique(sorted_df_perf['Name']))

    # number of wins
    dfff = df_ranks.rank(ascending=asc)
    if verbose:
        print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=asc).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets