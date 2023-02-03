from ast import literal_eval

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from charts import contour_plot_grid
from utils import get_dir


def column_to_grid(column, shape=None):
    if shape is None:
        s = int(np.sqrt(len(column) / 2))
        shape = (s, 2 * s)
    return np.array(column.apply(lambda values: literal_eval(values)[0])).reshape(shape)


def get_gradients(df, column, spacing=np.pi / 20):
    return np.gradient(column_to_grid(df[column]), spacing, edge_order=1)


def get_gradient_length(df, column, spacing=np.pi / 20, norm_order=1):
    gradients = get_gradients(df, column, spacing)
    return np.linalg.norm(np.dstack(gradients), ord=norm_order, axis=-1)


def get_min_max(gradient):
    return np.min(gradient), np.max(gradient)


def gradient_length_plot(csv_path, csv_path_expected, column_expected='qaoa', inverse=False,
                         save=False, norm_order=1):
    df = pd.read_csv(csv_path)
    df_expected = pd.read_csv(csv_path_expected)
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    alg_list = []
    shot_list = []
    length_list_var = []
    length_list_avg = []
    exponent = -1 if inverse else 1
    for column in columns:
        alg, shots = column.rsplit('_', 1)
        alg_list.append(alg)
        shot_list.append(int(shots))
        length_list_var.append(np.var(get_gradient_length(df, column, norm_order=norm_order)) ** exponent)
        length_list_avg.append(np.average(get_gradient_length(df, column, norm_order=norm_order)) ** exponent)

    df_new = pd.DataFrame(
        {'algorithm': alg_list, 'shots': shot_list, 'gradient_length_avg': length_list_avg,
         'gradient_length_var': length_list_var})

    path = get_dir(csv_path)
    gradient_sim_avg = np.average(get_gradient_length(df_expected, column_expected, norm_order=norm_order)) ** exponent
    gradient_sim_var = np.var(get_gradient_length(df_expected, column_expected, norm_order=norm_order)) ** exponent
    df_new.loc[len(df_new.index)] = ['simulator', '-', gradient_sim_avg, gradient_sim_var]
    df_new['gradient_length_avg_norm_sim'] = df_new['gradient_length_avg'] / gradient_sim_avg
    df_new['gradient_length_var_norm_sim'] = df_new['gradient_length_var'] / gradient_sim_var
    df_new.to_csv(f'{path}/gradient_length.csv')
    plt.rc('font', size=15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_new.set_index('shots').groupby('algorithm')['gradient_length_avg'].plot(ax=ax, x='shots', marker='o',
                                                                               legend=True,
                                                                               title='Average gradient size')
    plt.axhline(y=gradient_sim_avg, color='red', linestyle='--', label='simulator')
    ax.set_ylim(ymin=0)
    plt.legend()
    plt.show()
    if save:
        fig.savefig(f'{path}/gradient_length_avg.pdf')

    plt.rc('font', size=15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_new.set_index('shots').groupby('algorithm')['gradient_length_var'].plot(ax=ax, x='shots', marker='o',
                                                                               legend=True,
                                                                               title='Variance of gradient size')
    plt.axhline(y=gradient_sim_var, color='red', linestyle='--', label='simulator')
    ax.set_ylim(ymin=0)
    plt.legend()
    plt.show()
    if save:
        fig.savefig(f'{path}/gradient_length_var.pdf')


def gradient_std_plot(csv_path, csv_path_expected, column_expected='qaoa', inverse=False, parameter=None, mode='std',
                      save=False):
    df = pd.read_csv(csv_path)
    df_expected = pd.read_csv(csv_path_expected)
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    alg_list = []
    shot_list = []
    value_list_std = []
    value_list_avg = []
    exponent = -1 if inverse else 1
    if parameter is None:
        parameter = slice(None)
    for column in columns:
        alg, shots = column.rsplit('_', 1)
        alg_list.append(alg)
        shot_list.append(int(shots))
        value_list_std.append(np.std(get_gradients(df, column)[parameter]) ** exponent)
        value_list_avg.append(np.average(np.abs(get_gradients(df, column)[parameter])) ** exponent)

    name = f'{mode}_gradient'
    df_new = pd.DataFrame(
        {'algorithm': alg_list, 'shots': shot_list, 'std_gradient': value_list_std, 'avg_gradient': value_list_avg})

    path = get_dir(csv_path)
    df_new.to_csv(f'{path}/gradients.csv')
    plt.rc('font', size=15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_new.set_index('shots').groupby('algorithm')['std_gradient'].plot(ax=ax, x='shots', marker='o', legend=True,
                                                                        title='std_gradient')
    gradient_sim_std = np.std(get_gradients(df_expected, column_expected)[parameter]) ** exponent
    plt.axhline(y=gradient_sim_std, color='red', linestyle='--', label='simulator')
    ax.set_ylim(ymin=0)
    plt.legend()
    plt.show()
    if save:
        fig.savefig(f'{path}/gradient_std.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_new.set_index('shots').groupby('algorithm')['avg_gradient'].plot(ax=ax, x='shots', marker='o', legend=True,
                                                                        title='avg_gradient')
    gradient_sim_avg = np.average(np.abs(get_gradients(df_expected, column_expected)[parameter])) ** exponent
    plt.axhline(y=gradient_sim_avg, color='red', linestyle='--', label='simulator')
    ax.set_ylim(ymin=0)
    plt.legend()
    plt.show()
    if save:
        fig.savefig(f'{path}/gradient_avg.pdf')


def cosine_similarity(a1: np.ndarray, a2: np.ndarray):
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))


def cosine_similarity_gradients(df, column, df_expected, column_expected):
    gradients = get_gradients(df, column)
    gradients_expected = get_gradients(df_expected, column_expected)
    gradients_flat = np.stack(gradients).flat
    gradients_expected_flat = np.stack(gradients_expected).flat
    return cosine_similarity(gradients_flat, gradients_expected_flat)


def avg_cosine_similarity_gradients(df, column, df_expected, column_expected):
    gradients = np.dstack(get_gradients(df, column))
    gradients_expected = np.dstack(get_gradients(df_expected, column_expected))
    gradients = gradients.reshape(gradients.shape[0] * gradients.shape[1], gradients.shape[2])
    gradients_expected = gradients_expected.reshape(gradients_expected.shape[0] * gradients_expected.shape[1],
                                                    gradients_expected.shape[2])

    sim = []
    for g, g_e in zip(gradients, gradients_expected):
        sim.append(cosine_similarity(g, g_e))

    return np.average(sim)


def weighted_cosine_similarity_gradients(df, column, df_expected, column_expected):
    gradients = np.dstack(get_gradients(df, column))
    gradients_expected = np.dstack(get_gradients(df_expected, column_expected))
    gradients = gradients.reshape(gradients.shape[0] * gradients.shape[1], gradients.shape[2])
    gradients_expected = gradients_expected.reshape(gradients_expected.shape[0] * gradients_expected.shape[1],
                                                    gradients_expected.shape[2])

    sim = 0
    weights = 0
    for g, g_e in zip(gradients, gradients_expected):
        w = np.linalg.norm(g) * np.linalg.norm(g_e)
        sim += w * cosine_similarity(g, g_e)
        weights += w

    return sim / weights


def cosine_similarity_plot(csv_path_expected, csv_path, column_expected='qaoa', file_name='similarity'):
    df_expected = pd.read_csv(csv_path_expected)
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    alg_list = []
    shot_list = []
    similarity_list = []
    avg_similarity_list = []
    weighted_similarity_list = []
    for column in columns:
        alg, shots = column.rsplit('_', 1)
        alg_list.append(alg)
        shot_list.append(int(shots))
        similarity = cosine_similarity_gradients(df, column, df_expected, column_expected)
        avg_similarity = avg_cosine_similarity_gradients(df, column, df_expected, column_expected)
        weighted_similarity = weighted_cosine_similarity_gradients(df, column, df_expected, column_expected)
        similarity_list.append(similarity)
        avg_similarity_list.append(avg_similarity)
        weighted_similarity_list.append(weighted_similarity)

    df = pd.DataFrame(
        {'algorithm': alg_list, 'shots': shot_list, 'similarity': similarity_list,
         'avg_similarity': avg_similarity_list, 'weighted_similarity': weighted_similarity_list})

    path = get_dir(csv_path)

    plt.rc('font', size=15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.set_index('shots').groupby('algorithm')['similarity'].plot(ax=ax, x='shots', marker='o',
                                                                  legend=True,
                                                                  title='Cosine Similarity of Gradients')
    fig.savefig(f'{path}/{file_name}.pdf')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.set_index('shots').groupby('algorithm')['avg_similarity'].plot(ax=ax, x='shots', marker='o',
                                                                      legend=True,
                                                                      title='Avg. Pairwise Cosine Similarity of Gradients')
    fig.savefig(f'{path}/{file_name}_avg.pdf')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.set_index('shots').groupby('algorithm')['weighted_similarity'].plot(ax=ax, x='shots', marker='o',
                                                                           legend=True,
                                                                           title='Weighted Cosine Similarity of Gradients')
    fig.savefig(f'{path}/{file_name}_weighted.pdf')
    plt.show()
    df.to_csv(f'{path}/similarity.csv')


def plot_gradient(grid, name, title=None, dir_path=None, v_min_max=None, cmap='seismic', colorbar=True, levels=50):
    fig = contour_plot_grid(grid, title, v_min_max, cmap, colorbar, levels)
    if dir_path is not None:
        if title is not None:
            fig.savefig(f'{dir_path}/gradients_{name}_{title}.pdf')
        else:
            fig.savefig(f'{dir_path}/gradients_{name}.pdf')


def plot_gradient_maps(csv_path, column, v_min_max=None, cmap='seismic', colorbar=True, levels=50):
    df = pd.read_csv(csv_path, index_col=0)
    gradients = get_gradients(df, column)
    path = get_dir(csv_path)
    plot_gradient(gradients[0], f'{column}', title='beta', dir_path=path, v_min_max=v_min_max, cmap=cmap,
                  colorbar=colorbar,
                  levels=levels)
    plot_gradient(gradients[1], f'{column}', title='gamma', dir_path=path, v_min_max=v_min_max, cmap=cmap,
                  colorbar=colorbar,
                  levels=levels)


def plot_gradient_length_map(csv_path, column, v_min_max=None, cmap='seismic', colorbar=True, levels=50, norm_order=1):
    df = pd.read_csv(csv_path, index_col=0)
    gradient_lengths = get_gradient_length(df, column, norm_order=norm_order)
    path = get_dir(csv_path)
    plot_gradient(gradient_lengths, column, title='gradient_length', dir_path=path, v_min_max=v_min_max, cmap=cmap,
                  colorbar=colorbar,
                  levels=levels)


def all_gradients(path_expected, path, column_expected='qaoa', cmap='Spectral', cmap_length='inferno', colorbar=True,
                  levels=50, norm_order=1):
    gradient_std_plot(path, path_expected, column_expected, save=True)
    gradient_length_plot(path, path_expected, column_expected, save=True, norm_order=norm_order)
    df_sim = pd.read_csv(path_expected, index_col=0)
    sim_gradient = get_gradients(df_sim, column_expected)
    v_min_max = get_min_max(sim_gradient)
    plot_gradient_maps(path_expected, column_expected, v_min_max, cmap, colorbar, levels)
    plot_gradient_maps(path, 'qaoa_10000', v_min_max, cmap, colorbar, levels)
    try:
        plot_gradient_maps(path, 'qaoa-short_10000', v_min_max, cmap, colorbar, levels)
    except KeyError:
        pass
    plot_gradient_maps(path, 'cut-qaoa_10000', v_min_max, cmap, colorbar, levels)
    sim_gradient_length = get_gradient_length(df_sim, column_expected, norm_order=norm_order)
    v_min_max = get_min_max(sim_gradient_length)
    plot_gradient_length_map(path_expected, column_expected, v_min_max, cmap_length, colorbar, levels,
                             norm_order=norm_order)
    plot_gradient_length_map(path, 'qaoa_10000', v_min_max, cmap_length, colorbar, levels, norm_order=norm_order)
    try:
        plot_gradient_length_map(path, 'qaoa-short_10000', v_min_max, cmap_length, colorbar, levels,
                                 norm_order=norm_order)
    except KeyError:
        pass
    plot_gradient_length_map(path, 'cut-qaoa_10000', v_min_max, cmap_length, colorbar, levels, norm_order=norm_order)
