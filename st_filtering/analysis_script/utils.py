import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu


class DrawParam:
    def __init__(self):
        self.titlesize = 16
        self.labelsize = 14
        self.ticksize = 12
        self.legendsize = 12
        self.boxline = 1.5
        self.markersize = 7
        self.dot_palette = ['#B72230', '#317CB7']
        self.boxprops = dict(facecolor="none", edgecolor="gray", zorder=0)
        self.whiskerprops = dict(color="gray", linestyle="-", zorder=-1)
        self.medianprops = dict(color="orange", zorder=1)
        self.capprops = dict(color="gray", zorder=-1)
        self.flierprops = dict(marker='o', markerfacecolor='none', markersize=self.markersize, markeredgecolor='gray')


param = DrawParam()


def get_all_pair_distance(adata, ligand, receptor, cell_distance_matrix):
    ligand_cell = adata[adata[:, ligand].X > 0].obs_names
    receptor_cell = adata[adata[:, receptor].X > 0].obs_names
    distance = cell_distance_matrix.loc[ligand_cell, receptor_cell].values
    distance = distance[distance > 0]
    return distance


def get_distance(adata, ligand, receptor, sender, receiver, pair_info, cell_distance_matrix):
    cell_pair = pair_info.query('sender_type == @sender and receiver_type == @receiver')

    #
    sender_cell = adata[cell_pair['cell_sender_id']]
    ligand_expressing_sender = sender_cell[sender_cell[:, ligand].X > 0].obs_names

    receiver_cell = adata[cell_pair['cell_receiver_id']]
    receptor_expressing_receiver = receiver_cell[receiver_cell[:, receptor].X > 0].obs_names

    distance = cell_distance_matrix.loc[ligand_expressing_sender, receptor_expressing_receiver].values
    distance = distance[distance > 0]
    return distance


def plot_lrpair_spatial(adata, pair_info, ligand, receptor, sender, receiver, ax, arrow=True, legend_loc='best'):
    #
    print(ligand, receptor, sender, receiver)
    sender_cell = adata[adata.obsm['meta']['cell_type'] == sender]
    receiver_cell = adata[adata.obsm['meta']['cell_type'] == receiver]
    sender_cell = sender_cell[sender_cell[:, ligand].X > 0].obs_names  # ligand expressing sender
    receiver_cell = receiver_cell[receiver_cell[:, receptor].X > 0].obs_names  # receptor expressing receiver
    other_cell = adata[~adata.obs_names.isin(sender_cell.tolist() + receiver_cell.tolist())].obs_names

    #
    meta = adata.obsm['meta'].copy()
    sender_meta = meta.loc[sender_cell]
    receiver_meta = meta.loc[receiver_cell]
    other_meta = meta.loc[other_cell]

    #
    sender_name = sender.replace('_', ' ')
    sender_name = sender_name.replace('.', '&')
    receiver_name = receiver.replace('_', ' ')
    receiver_name = receiver_name.replace('.', '&')
    sender_c = 'xkcd:windows blue'
    receiver_c = 'xkcd:warm purple'
    ax.scatter(sender_meta['x'], sender_meta['y'], c=sender_c, s=param.markersize, label=f'{ligand} of {sender_name}',
               zorder=2)
    ax.scatter(receiver_meta['x'], receiver_meta['y'], c=receiver_c, s=param.markersize,
               label=f'{receptor} of {receiver_name}', zorder=1)
    ax.scatter(other_meta['x'], other_meta['y'], c='lightgrey', s=param.markersize, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=param.legendsize, loc=legend_loc)

    # add arrows
    cell_pair = pair_info.query('sender_type == @sender and receiver_type == @receiver')
    cell_pair = cell_pair.query('cell_sender_id in @sender_cell and cell_receiver_id in @receiver_cell').reset_index(
        drop=True)
    start = meta.loc[cell_pair['cell_sender_id']][['x', 'y']].values
    end = meta.loc[cell_pair['cell_receiver_id']][['x', 'y']].values

    if arrow:
        for p1, p2 in zip(start, end):
            x = p1[0]
            y = p1[1]
            dx = p2[0] - x
            dy = p2[1] - y
            ax.arrow(x, y, dx, dy, width=0.001, head_width=0.2, head_length=0.15, fc='black', ec='black',
                     length_includes_head=True, zorder=10)


def plot_lrpair_vln(adata, cell_distance_matrix, pair_info, ligand, receptor, sender, receiver, ax):
    d1 = get_all_pair_distance(adata=adata, ligand=ligand, receptor=receptor, cell_distance_matrix=cell_distance_matrix)
    d2 = get_distance(adata=adata, ligand=ligand, receptor=receptor, sender=sender, receiver=receiver,
                      pair_info=pair_info, cell_distance_matrix=cell_distance_matrix)

    # draw
    sender_name = sender.replace('_', ' ')
    sender_name = sender_name.replace('.', '&')
    receiver_name = receiver.replace('_', ' ')
    receiver_name = receiver_name.replace('.', '&')
    df = pd.DataFrame.from_dict(
        {'All cell-cell pairs': d1, f'{sender_name} - {receiver_name}': d2},
        orient='index'
    ).T

    # palette = ['lightgray', 'xkcd:lighter purple']
    palette = ['lightgray', '#FFC000']
    sns.boxplot(df, ax=ax, width=0.3, linewidth=param.boxline, palette=palette, saturation=0.9)
    ax.tick_params(axis='both', which='major', labelsize=param.ticksize)
    ax.tick_params(axis='x', rotation=5)
    ax.set_ylabel('Euclidean distance', fontsize=param.labelsize)
    x1, x2 = 0, 1
    y = max(np.max(d1), np.max(d2))
    y += 0.05 * y
    h, col = 0, 'k'

    p_value = mannwhitneyu(d1, d2, alternative='greater')[1]
    if p_value == 0:
        p_value_text = "$P$ < {:.0e}".format(1e-200)
    elif p_value < 1e-100:
        p_value_text = "$P$ < {:.0e}".format(1e-100)
    elif p_value < 1e-10:
        p_value_text = "$P$ < {:.0e}".format(1e-10)
    else:
        p_value_text = "$P$ = {:.2e}".format(p_value)
    ax.text((x1 + x2) * .5, y, p_value_text, ha='center', va='bottom', color=col, fontsize=param.legendsize)
    ax.set_ylim(-0.05 * y, 1.1 * y)
    ax.set_title(f'{ligand} $\\rightarrow$ {receptor}', fontsize=param.titlesize)


def plot_all_lrpair(df_lrdb, df_novel, ax, title, threshold=10):
    v1 = -np.log10(np.clip(df_lrdb['p_value'].values, 10 ** -threshold, 1))
    v2 = -np.log10(np.clip(df_novel['p_value'].values, 10 ** -threshold, 1))
    data = pd.DataFrame.from_dict({'LRDB': v1, 'Novel': v2}, orient='index').T

    # palette = ['xkcd:lighter purple', 'xkcd:light violet']
    palette = ['#2c7fb8', '#7fcdbb']
    sns.boxplot(data, width=0.3, ax=ax, linewidth=param.boxline, palette=palette, saturation=0.9)

    #
    yticks = [int(i) for i in ax.get_yticks().tolist()[1:-1]]
    ax.set_yticks(yticks, labels=yticks[:-1] + ['>{}'.format(yticks[-1])])
    ax.tick_params(axis='both', which='major', labelsize=param.ticksize)
    ax.set_ylabel('$-\log_{10}P$', fontsize=param.labelsize)
    y = max(np.max(v1), np.max(v2))
    ax.set_ylim(-0.05 * y, 1.1 * y)

    #
    ax.axhline(y=-np.log10(0.05), color='grey', linestyle='-.', zorder=10)
    ratio1 = (df_lrdb['p_value'] < 0.05).mean()
    ratio2 = (df_novel['p_value'] < 0.05).mean()
    # p=0.05
    ax.text(0.5, -np.log10(0.05) + 0.03 * y, '*\n$P=0.05$', fontsize=param.legendsize, ha='center')
    # num
    ax.text(0, 1.03 * y, '{}'.format(len(df_lrdb)), fontsize=param.legendsize, ha='center')
    ax.text(1, 1.03 * y, '{}'.format(len(df_novel)), fontsize=param.legendsize, ha='center')
    # ratio
    ax.text(0, 0.5 * y, '{:.2%} *'.format(ratio1), fontsize=param.legendsize, ha='center')
    ax.text(1, 0.5 * y, '{:.2%} *'.format(ratio2), fontsize=param.legendsize, ha='center')
    ax.set_title(title, fontsize=param.titlesize)



