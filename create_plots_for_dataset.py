import argparse
import logging
import re
import shutil
import sys
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, colors
from matplotlib.cm import ScalarMappable, Spectral
from matplotlib.colors import BoundaryNorm

from eval_param_maps import get_v_min_max
from eval_qaoa_run import get_appr_ratio_df, get_qaoa_execution, aggregate_counts, get_distribution_chart
from eval_util import average
from experiment_complete_eval import get_all_metrics_df, grid_plot_df, create_combined_df, boxplot_df, correlation, \
    plot_scatter_with_regression2
from graphs import draw_graph_from_file

logger = logging.getLogger(__name__)


def iterate_exp_dirs(path=None, min_exp=None, max_exp=None):
    if path is None:
        path = Path('experiment_complete')
    for exp_dir in path.iterdir():
        if not exp_dir.is_dir() or not re.match('^\d+$', exp_dir.name):
            continue
        if min_exp is not None and int(exp_dir.name) < min_exp:
            continue
        if max_exp is not None and int(exp_dir.name) > max_exp:
            continue
        sim_path = None
        qpu_path = None
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name.find('simulator') > -1:
                sim_path = run_dir
            else:
                qpu_path = run_dir
        yield exp_dir, sim_path, qpu_path


def load_metrics(path_dataset, relabel=None):
    metric_df = get_all_metrics_df(path_dataset)
    metric_df_f = metric_df.copy()
    # metric_df_f = metric_df[metric_df['algorithm'] != 'qaoa'].copy()
    relabel_algorithms(metric_df_f, relabel)
    return metric_df_f


def metric_plot(df, metric, ylabel, hue_order=None, figsize=(5, 4), markers=None):
    if markers is None:
        markers = ['o', 'v', 'p']
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.lineplot(ax=ax, data=df[df['metric'] == metric], x='shots', y='value', hue='algorithm', style='algorithm',
                 hue_order=hue_order, markers=markers, dashes=False)
    ax.set_ylabel(ylabel)
    return fig, ax


def add_sim_line(ax, path, metric):
    sim_val = pd.read_csv(path).query('algorithm =="simulator"').iloc[0][metric]
    ax.axhline(y=sim_val, color='red', linestyle='--', label='simulator')
    ax.legend()
    return ax


def plot_metrics(metrics_df, exp_id, plot_path, qpu_path, ylabels, hue_order=None):
    example_df = metrics_df[metrics_df['exp_id'] == str(exp_id)].copy()
    if hue_order is None:
        hue_order = ['cut-qaoa', 'qaoa']

    metric = 'mean_absolute_error'
    fig, ax = metric_plot(example_df, metric, ylabels[metric], hue_order)
    ax.legend()
    fig.tight_layout()
    fig.savefig((plot_path / f'{exp_id}_exp1_MAD_to_sim.pdf').resolve())

    metric = 'max_mixed_diff'
    fig, ax = metric_plot(example_df, metric, ylabels[metric], hue_order)
    ax = add_sim_line(ax, qpu_path / 'param_map/max_mixed_diff.csv', metric)
    fig.tight_layout()
    fig.savefig((plot_path / f'{exp_id}_exp1_MAD_to_MMS.pdf').resolve())

    metric = 'gradient_length_avg'
    fig, ax = metric_plot(example_df, metric, ylabels[metric], hue_order)
    ax = add_sim_line(ax, qpu_path / 'param_map/gradient_length.csv', metric)
    fig.tight_layout()
    fig.savefig((plot_path / f'{exp_id}_exp1_gradient_length_avg.pdf').resolve())

    metric = 'gradient_length_var'
    fig, ax = metric_plot(example_df, metric, ylabels[metric], hue_order)
    ax = add_sim_line(ax, qpu_path / 'param_map/gradient_length.csv', metric)
    fig.tight_layout()
    fig.savefig((plot_path / f'{exp_id}_exp1_gradient_length_var.pdf').resolve())

    metric = 'avg_similarity'
    fig, ax = metric_plot(example_df, metric, ylabels[metric], hue_order)
    ax.legend()
    fig.tight_layout()
    fig.savefig((plot_path / f'{exp_id}_exp1_similarity_avg.pdf').resolve())


def plot_shot_distribution(qpu_path, plot_path, exp_id, include_random=True, normalize=True, relabel=None, order=None):
    config, graph, result, execution_list = get_qaoa_execution(str((qpu_path / 'qaoa_execution').resolve()))
    counts = aggregate_counts(execution_list)
    if relabel is not None:
        relabeled_counts = {}
        for key, value in counts.items():
            relabeled_counts[relabel.get(key, key)] = value
        counts = relabeled_counts

    get_distribution_chart(graph, include_random=include_random, normalize=normalize,
                           path=(plot_path / f'{exp_id}_exp2_shot_distribution.pdf'), show=False, order=order,
                           **counts)


def add_config(exp_id, path, qpu_path):
    shutil.copy(qpu_path / 'config.json', path / f'{exp_id}_config.json')


def get_expectation_values(path, min_exp=13, max_exp=None, exclude_ids=None, relabel=None):
    exp_df = create_combined_df(path, min_exp=min_exp, max_exp=max_exp, config=None,
                                csv_files_to_collect={'': 'qaoa_execution/expectation.csv'}, add_config=True)
    exp_df_f = exp_df[exp_df['value_type'] != 'expectation'].copy()
    if exclude_ids is not None:
        exp_df_f = exp_df_f[~exp_df_f['exp_id'].isin(exclude_ids)]

    relabel_algorithms(exp_df_f, relabel)

    return exp_df_f


def get_approximation_ratio(path, min_exp=13, max_exp=None, exclude_ids=None, relabel=None):
    appr_df = get_appr_ratio_df(path, min_exp=min_exp, max_exp=max_exp, config=None, include_random=False)
    if exclude_ids is not None:
        appr_df = appr_df[~appr_df['exp_id'].isin(exclude_ids)]
    relabel_algorithms(appr_df, relabel)

    return appr_df


def relabel_algorithms(df, relabel):
    if relabel is not None:
        for old_label, new_label in relabel.items():
            df['algorithm'] = df['algorithm'].apply(
                lambda name: re.sub('^' + old_label + '$', new_label, name, 1))


def plot_func(*args, **kwargs):
    row_numbers = args[0].shape[0]
    shape = (int(np.sqrt(row_numbers / 2)), 2 * int(np.sqrt(row_numbers / 2)))
    grid = np.array(args[0].to_list()).reshape(shape)
    args = list(args)
    args[0] = grid
    kwargs.pop('color')
    cb = plt.contourf(*args, **kwargs)
    for c in cb.collections:
        c.set_edgecolor("face")
    return cb


def truncate_colormap(cmap, minval_qpu, maxval_qpu, minval_sim, maxval_sim, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    range_sim = maxval_sim - minval_sim
    minval = (minval_qpu - minval_sim) / range_sim
    maxval = (maxval_qpu - minval_sim) / range_sim
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_param_maps(exp_id, plot_path, sim_path, qpu_path, cmap=Spectral, levels=100, cols=None, relabel=None):
    v_min, v_max = get_v_min_max(sim_path / 'parameter_map.csv',
                                 'qaoa')

    if cols is None:
        cols = ['qaoa-short_1000', 'cut-qaoa_1000', 'qaoa-short_10000', 'cut-qaoa_10000']

    df_param_maps = pd.read_csv(qpu_path / 'param_map/parameter_map.csv', index_col=0)
    df_param_maps['parameters'] = df_param_maps['parameters'].apply(lambda x: literal_eval(x))
    df_param_maps['beta'] = df_param_maps['parameters'].apply(lambda p: p[0])
    df_param_maps['gamma'] = df_param_maps['parameters'].apply(lambda p: p[1])
    average(df_param_maps, cols)
    cols_and_params = [f'{c}_avg' for c in cols]
    value_vars = cols_and_params
    cols_and_params.extend(['beta', 'gamma'])
    df_params_reduced = df_param_maps[cols_and_params].copy()
    df_params_reduced = df_params_reduced.melt(id_vars=['beta', 'gamma'], value_vars=value_vars)
    df_params_reduced['algorithm'] = df_params_reduced['variable'].apply(lambda s: s.split('_')[0])
    relabel_algorithms(df_params_reduced, relabel)
    df_params_reduced['shots'] = df_params_reduced['variable'].apply(lambda s: int(s.split('_')[1]))

    min_qpu, max_qpu = df_params_reduced['value'].min(), df_params_reduced['value'].max()
    with sns.axes_style("ticks"):
        g = sns.FacetGrid(data=df_params_reduced, col='shots', row='algorithm', margin_titles=True, despine=False)
        g.map(plot_func, 'value', cmap=cmap, levels=levels, vmin=v_min, vmax=v_max, extent=(0, 2 * np.pi, 0, np.pi))
        g.set_xlabels(r'$\gamma$')
        g.set_ylabels(r'$\beta$')
        g.set(xticks=np.arange(0, 2 * np.pi + 0.01, np.pi), xticklabels=['$0$', r'$\pi$', r'$2\pi$'])
        g.set(yticks=np.arange(0, np.pi + 0.01, np.pi / 2), yticklabels=['$0$', r'$\dfrac{\pi}{2}$', r'$\pi$'])
        min_y = np.infty
        max_y = -np.infty
        for name, ax in g.axes_dict.items():
            for i in range(2):
                min_y = min(min_y, ax.get_position().get_points()[i, 1])
                max_y = max(max_y, ax.get_position().get_points()[i, 1])
        cbar_ax = g.fig.add_axes([1.015, min_y, 0.02, max_y - min_y])
        t_cmap = truncate_colormap(cmap, min_qpu, max_qpu, v_min, v_max)
        norm = BoundaryNorm(np.linspace(min_qpu, max_qpu, levels), t_cmap.N)
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=t_cmap), cax=cbar_ax)
        cbar.solids.set_edgecolor("face")
        g.set_titles(row_template='{row_name}', col_template='{col_name} shots')
    g.savefig(plot_path / f'{exp_id}_exp1_param_maps.pdf')


def plot_all_metrics(plot_path, metric_df, ylabels, markers, hue_order=None):
    ylims = {
        'mean_absolute_error': (0.0, 1.1),
        'mean_absolute_error_norm': (0.0, 0.175),
        'max_mixed_diff': (0, 1.1),
        'max_mixed_diff_norm_sim': (0, 1.1),
        'gradient_length_avg': (0, 5),
        'gradient_length_avg_norm_sim': (0, 1.5),
        'gradient_length_var': (0, 5),
        'gradient_length_var_norm_sim': (0, 1.2),
        'avg_similarity': (-0.2, 0.95)
    }
    xlim = (0, 10000)

    row_order = ['mean_absolute_error', 'max_mixed_diff_norm_sim', 'gradient_length_avg_norm_sim',
                 'gradient_length_var_norm_sim', 'avg_similarity']

    grid = grid_plot_df(metric_df, hue_order=hue_order, ylims=ylims, ylabels=ylabels, xlim=xlim,
                        row_order=row_order, height=3, style='algorithm', markers=markers, dashes=False, font_size=20,
                        aspect=1.25, facet_kws={'sharey': 'row', 'margin_titles': True, 'legend_out': False,
                                                'gridspec_kws': {'wspace': 0.15, 'hspace': 0.15}})
    grid.axes[0, 0].legend()
    plot_path.mkdir(parents=True, exist_ok=True)
    grid.savefig((plot_path / 'exp1_all.pdf').resolve())


def get_correlation_df(metric_df, appr_df, exp_df):
    exp_df_random = exp_df[(exp_df['algorithm'] == 'random') & (exp_df['value_type'] == 'expectation_norm')]
    appr_with_random = appr_df.merge(exp_df_random[['exp_id', 'value']], on=['exp_id'])
    appr_with_random['expect_ratio_rand'] = appr_with_random['expect_ratio'] - appr_with_random['value']
    appr_with_random['appr_ratio_rand'] = appr_with_random['appr_ratio'] - appr_with_random['value']
    appr_with_random = appr_with_random.rename(columns={"value": "random"})

    df = metric_df[metric_df['shots'] == 10000].copy()
    df_merge = df.merge(appr_with_random.drop(columns=['number_of_edges']), how='inner', on=['exp_id', 'algorithm'])

    df_melt = df_merge.melt(
        id_vars=['exp_id', 'algorithm', 'graph_size_x', 'backend_x', 'number_of_edges', 'metric', 'value'],
        value_vars=['obj_diff', 'appr_ratio', 'expect_ratio', 'expect_ratio_rand', 'appr_ratio_rand'],
        var_name="obj_metric", value_name="obj_value")
    return df_melt


def plot_correlation(plot_path, correlation_df, markers, exp2_metric='expect_ratio', columns=None, exclude_ids=None,
                     ylabel=None):
    df_filter = correlation_df[(correlation_df['obj_metric'] == exp2_metric)].copy()

    if columns is not None:
        df_filter = df_filter[df_filter['metric'].isin(columns)]

    if exclude_ids is not None:
        df_filter = df_filter[~df_filter['exp_id'].isin(exclude_ids)]

    cor_grid = plot_scatter_with_regression2(df_filter, x='value', y='obj_value', col='metric', hue='algorithm',
                                             col_order=columns, markers=markers, sharey=True, height=4.5, aspect=0.9,
                                             alpha=0.5, ylabel=ylabel, show=False)

    cor_grid.savefig((plot_path / f'correlation_{exp2_metric}.pdf').resolve())


def console_arguments():
    parser = argparse.ArgumentParser(description='Qiskit runtime experiment')
    parser.add_argument('-d', '--path-dataset', type=str, nargs='?', default='experiment_complete',
                        help='path of the dataset')
    parser.add_argument('-t', '--path-target', type=str, nargs='?', default='plots',
                        help='path to store the plots')

    return parser.parse_args()


def main():
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')
    args = console_arguments()
    path_dataset = Path(args.path_dataset)
    path_target = Path(args.path_target)

    relabel_map = {
        'cut-qaoa': 'cut-QAOA',
        'qaoa-short': 'QAOA (par.)',
        'cut_qaoa': 'cut-QAOA',
        'qaoa_short': 'QAOA (par.)',
        'qaoa': 'QAOA (seq.)',
    }

    order = ['cut-QAOA', 'QAOA (par.)', 'QAOA (seq.)']
    order_with_random = ['cut-QAOA', 'QAOA (par.)', 'QAOA (seq.)', 'random']

    cols = ['qaoa_1000', 'qaoa-short_1000', 'cut-qaoa_1000', 'qaoa_10000', 'qaoa-short_10000', 'cut-qaoa_10000']

    corr_cols = ['mean_absolute_error',
                 'max_mixed_diff_norm_sim',
                 'gradient_length_avg_norm_sim',
                 'gradient_length_var_norm_sim',
                 'avg_similarity']

    metrics = load_metrics(path_dataset, relabel=relabel_map)

    markers = ['o', 'v', 'p']

    exclude_ids = ['19', '21', '30']

    ylabels = {
        'mean_absolute_error': 'MAD to simulator',
        'mean_absolute_error_norm': 'Norm. MAD to simulator',
        'max_mixed_diff': 'MAD to MMS',
        'max_mixed_diff_norm_sim': 'Norm. MAD to MMS',
        'gradient_length_avg': 'Average gradient size',
        'gradient_length_avg_norm_sim': 'Norm. avg. grad. size',
        'gradient_length_var': 'Variance of gradient size',
        'gradient_length_var_norm_sim': 'Norm. var. of grad. size',
        'avg_similarity': 'Avg. pairwise cos. sim.'
    }

    context_boxplots = {'font.size': 10.0,
                        'axes.labelsize': 20,
                        'axes.titlesize': 20,
                        'xtick.labelsize': 20,
                        'ytick.labelsize': 20,
                        'legend.fontsize': 20,
                        'legend.title_fontsize': None,
                        'axes.linewidth': 0.8,
                        'grid.linewidth': 0.8,
                        'lines.linewidth': 1.5,
                        'lines.markersize': 6.0,
                        'patch.linewidth': 1.0,
                        'xtick.major.width': 0.8,
                        'ytick.major.width': 0.8,
                        'xtick.minor.width': 0.6,
                        'ytick.minor.width': 0.6,
                        'xtick.major.size': 3.5,
                        'ytick.major.size': 3.5,
                        'xtick.minor.size': 2.0,
                        'ytick.minor.size': 2.0}

    plot_all_metrics(path_target, metrics[metrics['graph_size'] != 14].copy(), ylabels, markers,
                     hue_order=order)

    exp_df = get_expectation_values(path_dataset, relabel=relabel_map, exclude_ids=exclude_ids)
    bp_exp = boxplot_df(exp_df[exp_df['graph_size'] != 14].copy(), x='algorithm', y='value', col='graph_size',
                        order=order_with_random,
                        ylabel='Expectation ratio', xlabel='Variant', median_fontsize=16, context=context_boxplots,
                        xlabel_rotation=20)
    bp_exp.savefig((path_target / 'exp2_expectation_boxplot.pdf').resolve())
    appr_df = get_approximation_ratio(path_dataset, relabel=relabel_map, exclude_ids=exclude_ids)
    bp_appr = boxplot_df(appr_df[appr_df['graph_size'] != 14].copy(), x='algorithm', y='appr_ratio', col='graph_size',
                         order=order,
                         ylabel='Approximation ratio', xlabel='Variant', median_fontsize=16, context=context_boxplots,
                         xlabel_rotation=20)

    bp_appr.savefig((path_target / 'exp2_appr_ratio_boxplot.pdf').resolve())

    corr_df = get_correlation_df(metrics, appr_df, exp_df)
    plot_correlation(path_target, corr_df, markers, exp2_metric='expect_ratio', columns=corr_cols,
                     exclude_ids=exclude_ids)
    plot_correlation(path_target, corr_df, markers, exp2_metric='expect_ratio_rand', columns=corr_cols,
                     exclude_ids=exclude_ids, ylabel='Expectation ratio increase\ncompared to MMS')

    for exp_dir, sim_path, qpu_path in iterate_exp_dirs(path_dataset):
        exp_id = exp_dir.name
        print(exp_id)
        path = path_target / exp_id
        path.mkdir(parents=True, exist_ok=True)
        draw_graph_from_file((qpu_path / 'graph.txt').resolve(), path / f'{exp_id}_graph.pdf', show=False)
        plot_metrics(metrics, exp_id, path, qpu_path, ylabels, hue_order=order)
        add_config(exp_id, path, qpu_path)
        plot_param_maps(exp_id, path, sim_path, qpu_path, cols=cols, relabel=relabel_map)
        plot_shot_distribution(qpu_path, path, exp_id, relabel=relabel_map, order=order_with_random)
        plt.close('all')


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    s_handler = logging.StreamHandler(sys.stdout)
    main()
