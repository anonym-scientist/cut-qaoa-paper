import json
import re
from collections import ChainMap
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import stats

from charts import contour_plot_grid
from eval_gradients import all_gradients, cosine_similarity_plot
from eval_param_maps import all_errors, max_mixed_diff_plot
from eval_qaoa_run import draw_chart, add_median_labels, get_appr_ratio_df, get_qaoa_execution
from eval_util import get_max_qaoa, store_config_info, add_config_for_exp_id, average
from circuit_cutting.execute import ExecutorResults

ALL_CSV_FILES = {
    'error': 'param_map/errors.csv',
    'gradients': 'param_map/gradient_length.csv',
    'max_mixed_diff': 'param_map/max_mixed_diff.csv',
    'grad_similarity': 'param_map/similarity.csv',
    'qaoa': 'qaoa_execution/expectation.csv'

}


def find_csv_files(path, min_exp=None, max_exp=None, config=None, csv_files_to_collect=None):
    if csv_files_to_collect is None:
        csv_files_to_collect = ALL_CSV_FILES
    sheet_csv_map = {}
    if not isinstance(path, Path):
        path = Path(path)
    for exp_dir in path.iterdir():
        if not exp_dir.is_dir() or not re.match('^\d+$', exp_dir.name):
            continue
        if min_exp is not None and int(exp_dir.name) < min_exp:
            continue
        if max_exp is not None and int(exp_dir.name) > max_exp:
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name.find('simulator') > -1:
                continue
            if config is not None:
                with open((run_dir / 'config.json').resolve()) as f:
                    config_json = json.load(f)

                continue_outer = False
                for key, value in config.items():
                    if config_json[key] != value:
                        continue_outer = True
                        break
                if continue_outer:
                    continue

            for name, csv_path in csv_files_to_collect.items():
                if name == '':
                    sheet_csv_map[f'{exp_dir.name}'] = run_dir / csv_path
                else:
                    sheet_csv_map[f'{exp_dir.name}_{name}'] = run_dir / csv_path
    return sheet_csv_map


def aggregate_dict_list(dict_list, func):
    agg_dict = {}
    for key in dict_list[0].keys():
        agg_dict[key] = func(np.array([d[key] for d in dict_list]))
    return agg_dict


def store_circuit_info(path):
    if isinstance(path, str):
        path = Path(path)

    exec_results = ExecutorResults.load_results((path / 'executor_results.json'))
    transpile_info = exec_results._transpilation_info

    transpile_info_copy = deepcopy(transpile_info)
    for shots, result_list in exec_results._results.items():
        info_list = transpile_info[str(shots)]
        for i, (result, info) in enumerate(zip(result_list, info_list)):
            transpile_info_copy[str(shots)][i]['width'] = info['width'] - result.results[0].header.memory_slots

    shot_list = list(transpile_info.keys())
    assert len(shot_list) == 2
    shots_cut = min(shot_list)
    shots = max(shot_list)
    transpile_info_list_cut = transpile_info_copy[shots_cut]
    transpile_info_list = transpile_info_copy[shots]

    mean_transpile_info_cut = aggregate_dict_list(transpile_info_list_cut, np.mean)
    median_transpile_info_cut = aggregate_dict_list(transpile_info_list_cut, np.median)
    mean_transpile_info_ = aggregate_dict_list(transpile_info_list, np.mean)
    median_transpile_info_ = aggregate_dict_list(transpile_info_list, np.median)

    print(mean_transpile_info_cut)
    print(median_transpile_info_cut)
    print(mean_transpile_info_)
    print(median_transpile_info_)


def eval_all(path_expected, path, column_expected='qaoa', csv_name_expected='parameter_map.csv',
             csv_name='param_map/parameter_map.csv', cmap='Spectral', levels=50):
    csv_path = f'{path}/{csv_name}'
    csv_path_expected = f'{path_expected}/{csv_name_expected}'
    all_errors(csv_path_expected, csv_path, column_expected=column_expected, cmap=cmap, levels=levels)
    all_gradients(csv_path_expected, csv_path, column_expected=column_expected, cmap=cmap, colorbar=True, levels=levels)
    max_mixed_diff_plot(csv_path_expected, f'{path}/param_map', column_expected=column_expected)
    cosine_similarity_plot(csv_path_expected, csv_path, column_expected=column_expected)
    draw_chart(f'{path}/qaoa_execution')
    store_config_info(path)


def eval_all_dirs(path=None, min_exp=None, max_exp=None):
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
            if run_dir.name.find('aer_simulator') > -1:
                sim_path = run_dir
            else:
                qpu_path = run_dir

        print(f'Eval {exp_dir.name}')
        eval_all(sim_path, qpu_path)
        plt.close('all')


def create_combined_df(path=None, min_exp=None, max_exp=None, config=None, csv_files_to_collect=None, add_config=False):
    if path is None:
        path = Path('experiment_complete')
    if csv_files_to_collect is None:
        csv_files_to_collect = {'': 'param_map/errors.csv'}
    csv_files = find_csv_files(path, min_exp, max_exp, config, csv_files_to_collect)
    df_list = []
    for exp_id, file in csv_files.items():
        df = pd.read_csv(file.resolve(), index_col=0)
        df['exp_id'] = [exp_id for _ in range(df.shape[0])]
        if add_config:
            df = add_config_for_exp_id(df, exp_id, path)
        df_list.append(df)
    merged_df = pd.concat(df_list)
    merged_df.reset_index(inplace=True, drop=True)
    if 'algorithm' in merged_df.columns:
        merged_df = merged_df[merged_df['algorithm'] != 'simulator'].copy()
    if 'shots' in merged_df.columns:
        merged_df.loc[:, 'shots'] = merged_df.loc[:, 'shots'].astype(int)
    return merged_df


def create_merged_plot(df, y, x='shots', hue='algorithm', ylim=None, path=None, **kwargs):
    plt.rc('font', size=15)
    fig, ax = plt.subplots()
    # Todo remove
    df = df[df['algorithm'] != 'simulator'].copy()
    df.loc[:, 'shots'] = df.loc[:, 'shots'].astype(int)
    sns.lineplot(ax=ax, data=df, x=x, y=y, hue=hue, **kwargs)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()
    if path is not None:
        fig.savefig(path)


def create_relplot(df, y, x='shots', hue='algorithm', col='graph_size', ylim=None, path=None, **kwargs):
    plt.rc('font', size=15)
    # Todo remove
    df = df[df['algorithm'] != 'simulator'].copy()
    df.loc[:, 'shots'] = df.loc[:, 'shots'].astype(int)
    grid = sns.relplot(data=df, x=x, y=y, hue=hue, col=col, kind='line', **kwargs)
    if ylim is not None:
        grid.set(ylim=ylim)
    plt.tight_layout()
    plt.show()
    # if path is not None:
    #     fig.savefig(path)


def all_merged_plots(path=None, min_exp=None, max_exp=None, config=None, lines=False, add_config=False,
                     errors_kwargs=None,
                     gradient_length_kwargs=None,
                     max_mixed_diff_kwargs=None, similarity_kwargs=None, **kwargs):
    if path is None:
        path = Path('experiment_complete')
    errors_kwargs = {} if errors_kwargs is None else errors_kwargs
    gradient_length_kwargs = {} if gradient_length_kwargs is None else gradient_length_kwargs
    max_mixed_diff_kwargs = {} if max_mixed_diff_kwargs is None else max_mixed_diff_kwargs
    similarity_kwargs = {} if similarity_kwargs is None else similarity_kwargs
    if lines:
        kwargs['estimator'] = None
        kwargs['units'] = 'exp_id'

    df_errors = create_combined_df(path, min_exp, max_exp, config,
                                   csv_files_to_collect={'': 'param_map/errors.csv'}, add_config=add_config)
    df_gradient_length = create_combined_df(path, min_exp, max_exp, config,
                                            csv_files_to_collect={'': 'param_map/gradient_length.csv'},
                                            add_config=add_config)
    df_max_mixed_diff = create_combined_df(path, min_exp, max_exp, config,
                                           csv_files_to_collect={'': 'param_map/max_mixed_diff.csv'},
                                           add_config=add_config)
    df_similarity = create_combined_df(path, min_exp, max_exp, config,
                                       csv_files_to_collect={'': 'param_map/similarity.csv'}, add_config=add_config)
    path = path / 'plots'
    if config is not None:
        for (k, v) in sorted(config.items()):
            path = path / f'{str(k)}_{str(v)}'
    if lines:
        path = path / 'lines'
    else:
        path = path / 'aggregated'
    path.mkdir(parents=True, exist_ok=True)
    create_merged_plot(df_errors, 'mean_absolute_error', path=path / 'errors_plot.pdf',
                       **ChainMap(errors_kwargs, kwargs))
    create_merged_plot(df_max_mixed_diff, 'max_mixed_diff', path=path / 'max_mixed_plot.pdf',
                       **ChainMap(max_mixed_diff_kwargs, kwargs))
    create_merged_plot(df_gradient_length, 'gradient_length_avg', path=path / 'gradient_length_avg_plot.pdf',
                       **ChainMap(gradient_length_kwargs, kwargs))
    create_merged_plot(df_gradient_length, 'gradient_length_var', path=path / 'gradient_length_var_plot.pdf',
                       **ChainMap(gradient_length_kwargs, kwargs))
    create_merged_plot(df_similarity, 'avg_similarity', path=path / 'avg_similarity_plot.pdf',
                       **ChainMap(similarity_kwargs, kwargs))


def grid_plot(path=None, min_exp=None, max_exp=None, config=None, lines=False, add_config=True, merge_qaoa=False,
              errors_kwargs=None,
              gradient_length_kwargs=None,
              max_mixed_diff_kwargs=None, similarity_kwargs=None, **kwargs):
    if path is None:
        path = Path('experiment_complete')
    errors_kwargs = {} if errors_kwargs is None else errors_kwargs
    gradient_length_kwargs = {} if gradient_length_kwargs is None else gradient_length_kwargs
    max_mixed_diff_kwargs = {} if max_mixed_diff_kwargs is None else max_mixed_diff_kwargs
    similarity_kwargs = {} if similarity_kwargs is None else similarity_kwargs
    if lines:
        kwargs['estimator'] = None
        kwargs['units'] = 'exp_id'

    df_errors = create_combined_df(path, min_exp, max_exp, config,
                                   csv_files_to_collect={'': 'param_map/errors.csv'}, add_config=add_config)
    df_gradient_length = create_combined_df(path, min_exp, max_exp, config,
                                            csv_files_to_collect={'': 'param_map/gradient_length.csv'},
                                            add_config=add_config)
    df_max_mixed_diff = create_combined_df(path, min_exp, max_exp, config,
                                           csv_files_to_collect={'': 'param_map/max_mixed_diff.csv'},
                                           add_config=add_config)
    df_similarity = create_combined_df(path, min_exp, max_exp, config,
                                       csv_files_to_collect={'': 'param_map/similarity.csv'}, add_config=add_config)

    df_errors = df_errors.melt(id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend'],
                               value_vars=['mean_absolute_error'], var_name="metric", value_name="value")
    df_gradient_length = df_gradient_length.melt(id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend'],
                                                 value_vars=['gradient_length_avg', 'gradient_length_var'],
                                                 var_name="metric", value_name="value")
    df_max_mixed_diff = df_max_mixed_diff.melt(id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend'],
                                               value_vars=['max_mixed_diff'], var_name="metric", value_name="value")
    df_similarity = df_similarity.melt(id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend'],
                                       value_vars=['avg_similarity'], var_name="metric", value_name="value")
    df = pd.concat([df_errors, df_gradient_length, df_max_mixed_diff, df_similarity])
    df = df[df['graph_size'] != 14].copy()
    print(df.shape)
    ylims = {
        'mean_absolute_error': (0.2, 1.1),
        'max_mixed_diff': (0, 1.1),
        'gradient_length_avg': (0, 5),
        'gradient_length_var': (0, 5),
        'avg_similarity': (-0.35, 0.95)
    }
    ylabels = {
        'mean_absolute_error': 'MAD to simulator',
        'max_mixed_diff': 'MAD to max. mix. state',
        'gradient_length_avg': 'Average gradient size',
        'gradient_length_var': 'Variance of gradient size',
        'avg_similarity': 'Avg. pairwise cos. sim.'
    }
    hue_order = ['cut-qaoa', 'qaoa']
    if merge_qaoa:
        df['algorithm'] = df['algorithm'].apply(lambda name: name.replace('qaoa-short', 'qaoa'))
    else:
        hue_order.append('qaoa-short')

    grid = grid_plot_df(df, hue_order, ylims, ylabels, **kwargs)
    grid.savefig((path / 'all.pdf').resolve())


def grid_plot_df(df, ylims=None, ylabels=None, font_size=20,
                 xlim=None, **kwargs):
    plt.rc('font', size=font_size)
    config = {'x': 'shots',
              'y': 'value',
              'hue': 'algorithm',
              'col': 'graph_size',
              'row': 'metric', 'kind': 'line',
              'facet_kws': {'sharey': 'row', 'margin_titles': True, 'legend_out': False},
              'errorbar': ('pi', 80),
              'row_order': ['mean_absolute_error', 'max_mixed_diff', 'gradient_length_avg', 'gradient_length_var',
                            'avg_similarity'],
              'aspect': 1.25
              }

    grid = sns.relplot(data=df, **ChainMap(kwargs, config))
    grid.set_titles(col_template='Number of nodes: {col_name}', row_template='')
    if ylims is None:
        ylims = {}
        ylabels = {}
    for (y, x), ax in grid.axes_dict.items():
        if y in ylims.keys():
            ax.set_ylim(ylims[y])
            ax.set_ylabel(ylabels[y])
        if xlim is not None:
            ax.set_xlim(xlim)
    for ax in grid.axes[:, 0]:
        ax.get_yaxis().set_label_coords(-0.2, 0.5)
    return grid


def edges_gird_plot(path=None, min_exp=None, max_exp=None, config=None, lines=False, add_config=True,
                    errors_kwargs=None,
                    gradient_length_kwargs=None,
                    max_mixed_diff_kwargs=None, similarity_kwargs=None, **kwargs):
    if path is None:
        path = Path('experiment_complete')
    errors_kwargs = {} if errors_kwargs is None else errors_kwargs
    gradient_length_kwargs = {} if gradient_length_kwargs is None else gradient_length_kwargs
    max_mixed_diff_kwargs = {} if max_mixed_diff_kwargs is None else max_mixed_diff_kwargs
    similarity_kwargs = {} if similarity_kwargs is None else similarity_kwargs
    if lines:
        kwargs['estimator'] = None
        kwargs['units'] = 'exp_id'

    df_errors = create_combined_df(path, min_exp, max_exp, config,
                                   csv_files_to_collect={'': 'param_map/errors.csv'}, add_config=add_config)
    print(df_errors)
    df_gradient_length = create_combined_df(path, min_exp, max_exp, config,
                                            csv_files_to_collect={'': 'param_map/gradient_length.csv'},
                                            add_config=add_config)
    df_max_mixed_diff = create_combined_df(path, min_exp, max_exp, config,
                                           csv_files_to_collect={'': 'param_map/max_mixed_diff.csv'},
                                           add_config=add_config)
    df_similarity = create_combined_df(path, min_exp, max_exp, config,
                                       csv_files_to_collect={'': 'param_map/similarity.csv'}, add_config=add_config)

    df_errors = df_errors.melt(id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
                               value_vars=['mean_absolute_error'], var_name="metric", value_name="value")
    df_gradient_length = df_gradient_length.melt(
        id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
        value_vars=['gradient_length_avg', 'gradient_length_var'],
        var_name="metric", value_name="value")
    df_max_mixed_diff = df_max_mixed_diff.melt(
        id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
        value_vars=['max_mixed_diff'], var_name="metric", value_name="value")
    df_similarity = df_similarity.melt(
        id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
        value_vars=['avg_similarity'], var_name="metric", value_name="value")
    df = pd.concat([df_errors, df_gradient_length, df_max_mixed_diff, df_similarity])
    df = df[df['shots'] == 10000].copy()
    df = df[df['graph_size'] != 14].copy()
    print(df.shape)
    print(df)
    plt.rc('font', size=20)
    ylims = {
        'mean_absolute_error': (0.2, 1.1),
        'max_mixed_diff': (0, 1.1),
        'gradient_length_avg': (0, 5),
        'gradient_length_var': (0, 5),
        'avg_similarity': (-0.35, 0.95)
    }
    grid = sns.relplot(data=df, x='number_of_edges', y='value', hue='algorithm', row='metric', kind='line',
                       size='graph_size',
                       facet_kws={'sharey': 'row', 'margin_titles': True, 'legend_out': True}, errorbar=None,
                       row_order=['mean_absolute_error', 'max_mixed_diff', 'gradient_length_avg', 'gradient_length_var',
                                  'avg_similarity'],
                       hue_order=['cut-qaoa', 'qaoa', 'qaoa-short'], aspect=1.25, **kwargs)
    grid.set_titles(col_template='Number of nodes: {col_name}', row_template='')
    for y, ax in grid.axes_dict.items():
        if y in ylims.keys():
            # ax.set_ylim(ylims[y])
            ax.set_ylabel(y)
    # grid.savefig((path / 'all.pdf').resolve())
    plt.show()


def all_merged_plots_and_lines(path=None, min_exp=None, max_exp=None, config=None, errorbar=('pi', 80), **kwargs):
    errors_kwargs = {'ylim': (0.2, 1.1), 'style': 'graph_size'}
    max_mixed_diff_kwargs = {'ylim': (0, 1.1)}
    gradient_length_kwargs = {'ylim': (0, 5)}
    similarity_kwargs = {'ylim': (-0.35, 0.95)}

    all_merged_plots(path=path, min_exp=min_exp, max_exp=max_exp, config=config, errorbar=errorbar,
                     gradient_length_kwargs=gradient_length_kwargs,
                     max_mixed_diff_kwargs=max_mixed_diff_kwargs, errors_kwargs=errors_kwargs,
                     similarity_kwargs=similarity_kwargs, add_config=True, **kwargs)
    all_merged_plots(path=path, min_exp=min_exp, max_exp=max_exp, config=config, lines=True,
                     gradient_length_kwargs=gradient_length_kwargs,
                     max_mixed_diff_kwargs=max_mixed_diff_kwargs,
                     errors_kwargs=errors_kwargs, similarity_kwargs=similarity_kwargs, add_config=True, **kwargs)


def get_all_metrics_df(path=None, min_exp=None, max_exp=None, config=None, add_config=True):
    if path is None:
        path = Path('experiment_complete')
    df_errors = create_combined_df(path, min_exp, max_exp, config,
                                   csv_files_to_collect={'': 'param_map/errors.csv'}, add_config=add_config)
    df_gradient_length = create_combined_df(path, min_exp, max_exp, config,
                                            csv_files_to_collect={'': 'param_map/gradient_length.csv'},
                                            add_config=add_config)
    df_max_mixed_diff = create_combined_df(path, min_exp, max_exp, config,
                                           csv_files_to_collect={'': 'param_map/max_mixed_diff.csv'},
                                           add_config=add_config)
    df_similarity = create_combined_df(path, min_exp, max_exp, config,
                                       csv_files_to_collect={'': 'param_map/similarity.csv'}, add_config=add_config)

    df_errors = df_errors.melt(id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
                               value_vars=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_error_norm',
                                           'mean_squared_error_norm'], var_name="metric",
                               value_name="value")
    df_gradient_length = df_gradient_length.melt(
        id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
        value_vars=['gradient_length_avg', 'gradient_length_var', 'gradient_length_avg_norm_sim',
                    'gradient_length_var_norm_sim'],
        var_name="metric", value_name="value")
    df_max_mixed_diff = df_max_mixed_diff.melt(
        id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
        value_vars=['max_mixed_diff', 'max_mixed_diff_norm_sim'], var_name="metric", value_name="value")
    df_similarity = df_similarity.melt(
        id_vars=['exp_id', 'algorithm', 'shots', 'graph_size', 'backend', 'number_of_edges'],
        value_vars=['similarity', 'avg_similarity', 'weighted_similarity'], var_name="metric", value_name="value")
    df = pd.concat([df_errors, df_gradient_length, df_max_mixed_diff, df_similarity])
    return df


def expectation_boxplot(path=None, min_exp=None, max_exp=None, config=None, add_config=True, merge_qaoa=False,
                        max_qaoa=False):
    if path is None:
        path = Path('experiment_complete')
    df = create_combined_df(path, min_exp, max_exp, config, csv_files_to_collect={'': 'qaoa_execution/expectation.csv'},
                            add_config=add_config)
    df = df[df['value_type'] != 'expectation'].copy()
    df = df[df['graph_size'] != 14].copy()
    order = ['cut_qaoa', 'qaoa', 'random']
    if merge_qaoa:
        df['algorithm'] = df['algorithm'].apply(lambda name: name.replace('qaoa_short', 'qaoa'))
    else:
        order.insert(2, 'qaoa_short')
    if max_qaoa:
        df = get_max_qaoa(df, group_columns=['exp_id', 'value_type'])
        order.insert(len(order) - 1, 'max_qaoa')
    bp = boxplot_df(df, x='algorithm', y='value', col='graph_size', order=order, ylabel='Expectation ratio',
                    xlabel='Variant')
    plt.tight_layout()
    bp.savefig((path / 'expectation_boxplot.pdf').resolve())
    plt.show()


def boxplot_df(df, x='algorithm', y='value', col='graph_size', order=None, ylabel=None, xlabel=None,
               median_fontsize=14, context=None, xlabel_rotation=None):
    with sns.plotting_context(context):
        bp = sns.catplot(df, x=x, y=y, order=order, kind='box', col=col, height=5)
        bp.set(ylabel=ylabel, xlabel=xlabel)
        for ax in bp.axes_dict.values():
            add_median_labels(ax, fontsize=median_fontsize)
            _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=xlabel_rotation)
            ax.xaxis.labelpad = 20
    return bp


def correlation(path=None, min_exp=None, max_exp=None, config=None, add_config=True, df_y=None, n_max=None,
                merge_qaoa=False, **kwargs):
    if path is None:
        path = Path('experiment_complete')

    if df_y is None:
        df_y = get_appr_ratio_df(path, min_exp, max_exp, config, include_random=False)
    df = get_all_metrics_df(path, min_exp, max_exp, config, add_config)
    df = df[df['shots'] == 10000].copy()

    df_y['algorithm'] = df_y['algorithm'].apply(lambda x: x.replace('_', '-'))
    df_y = df_y[df_y['graph_size'] != 14].copy()
    df_merge = df.merge(df_y, how='inner', on=['exp_id', 'algorithm'])
    df_melt = df_merge.melt(
        id_vars=['exp_id', 'algorithm', 'graph_size_x', 'backend_x', 'number_of_edges', 'metric', 'value'],
        value_vars=['obj_diff', 'appr_ratio', 'expect_ratio', 'expect_ratio_rand', 'appr_ratio_rand'],
        var_name="obj_metric", value_name="obj_value")
    if n_max is not None:
        df_max = df_melt[df_melt['obj_metric'] != 'obj_diff']
        df_max = df_max.groupby(['exp_id', 'algorithm', 'metric', 'obj_metric'], group_keys=False).apply(
            lambda x: x.sort_values(by='obj_value', ascending=False).head(n_max))
        df_min = df_melt[df_melt['obj_metric'] == 'obj_diff']
        df_min = df_min.groupby(['exp_id', 'algorithm', 'metric', 'obj_metric'], group_keys=False).apply(
            lambda x: x.sort_values(by='obj_value', ascending=True).head(n_max))
        df_melt = pd.concat([df_min, df_max])
    order = ['cut-qaoa', 'qaoa']
    if merge_qaoa:
        df_melt['algorithm'] = df_melt['algorithm'].apply(lambda name: name.replace('qaoa_short', 'qaoa'))
    else:
        order.append('qaoa-short')
    plot = sns.lmplot(df_melt, x='value', y='obj_value', hue=None, col='metric', row='obj_metric',
                      hue_order=order, facet_kws={'sharey': False, 'sharex': False}, **kwargs)
    filename = 'correlations.pdf' if n_max is None else f'correlations_{n_max}_best.pdf'
    plot.savefig((path / filename).resolve())
    plt.show()
    return df_melt
    ncol = len(df_melt['metric'].unique())
    nrow = len(df_melt['obj_metric'].unique())
    ualgo = df_melt['algorithm'].unique()

    height = 5
    aspect = 1
    figsize = (ncol * height * aspect, nrow * height)

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=False, sharey=False, figsize=figsize)
    for ax, (title, grp1) in zip(axes.flat, df_melt.groupby(['obj_metric', 'metric'])):
        ax.set_title(title)
        ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
        sns.scatterplot(ax=ax, x="value", y="obj_value", data=grp1, hue='algorithm', size='graph_size_x', alpha=0.33,
                        markers='.',
                        hue_order=['cut-qaoa', 'qaoa', 'qaoa-short'])
        # for algo in ualgo:
        #     grp2 = grp1[grp1["algorithm"] == algo]
        #     sns.regplot(x="value", y="obj_value", data=grp2, label=str(algo),
        #                 fit_reg=False, ax=ax)
        # ax.legend(title="Algorithm")
    for ax, (time, grp1) in zip(axes.flat, df_melt.groupby(['obj_metric', 'metric'])):
        sns.regplot(x="value", y="obj_value", data=grp1, ax=ax, scatter=False,
                    color="k", label="regression line")
    fig.tight_layout()
    plt.show()


def plot_scatter_with_regression(df, x, y, row, col, hue=None, hue_order=None, size=None, height=5, aspect=1,
                                 sharex=False, sharey=False, single_reg=True, reg_equation=False):
    ncol = len(df[col].unique())
    nrow = len(df[row].unique())
    figsize = (ncol * height * aspect, nrow * height)

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=sharex, sharey=sharey, figsize=figsize)

    for ax, (title, grp1) in zip(axes.flat, df.groupby([row, col])):
        slope, intercept, r_value, pv, se = stats.linregress(grp1[x], grp1[y])
        print(title, '$y=%3.7sx+%3.7s$' % (slope, intercept))
        label = '$y=%3.7sx+%3.7s$' % (slope, intercept) if reg_equation else None
        if hue is None or single_reg == True:
            sns.regplot(x=x, y=y, data=grp1, ax=ax, scatter=False,
                        color="k", label=label)
        else:
            if hue_order is None:
                hue_order = df[hue].unique()
            for h in hue_order:
                data = grp1[grp1[hue] == h]
                slope, intercept, r_value, pv, se = stats.linregress(data[x], data[y])
                sns.regplot(x=x, y=y, data=data, ax=ax, scatter=False, label='$y=%3.7sx+%3.7s$' % (slope, intercept))

    for ax, (title, grp1) in zip(axes.flat, df.groupby([row, col])):
        ax.set_title(title)
        ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
        sns.scatterplot(ax=ax, x=x, y=y, data=grp1, hue=hue, size=size, alpha=0.33,
                        markers='.',
                        hue_order=hue_order)
    h, l = ax.get_legend_handles_labels()
    figlegend = fig.legend(h, l, loc='center right')
    fig.tight_layout()
    plt.show()


def plot_scatter_with_regression2(df, x, y, row=None, col=None, hue=None, hue_order=None, col_order=None, height=5,
                                  aspect=1,
                                  sharex=False, sharey=False, legend_out=True, markers=['o', 'x'], alpha=0.45,
                                  ylabel=None, show=True, **kwargs):
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, hue_order=hue_order, col_order=col_order, sharex=sharex,
                      sharey=sharey, height=height,
                      aspect=aspect, legend_out=legend_out, hue_kws={'marker': markers}, gridspec_kws={'wspace': 0.1})
    g.map(sns.scatterplot, x, y, alpha=alpha, **kwargs)
    g._hue_var = None
    g.hue_names = None
    g.hue_kws.pop('marker')
    g.map(sns.regplot, x, y, scatter=False, color="k")
    g.set_titles(col_template='', row_template='')
    xlabels = {
        'mean_absolute_error': 'MAD to simulator',
        'mean_absolute_error_norm': 'Norm. MAD to simulator',
        'max_mixed_diff_norm_sim': 'Norm. MAD to MMS',
        'gradient_length_avg_norm_sim': 'Norm. average gradient size',
        'gradient_length_var_norm_sim': 'Norm. variance of gradient size',
        'avg_similarity': 'Avg. pairwise cos. similarity'
    }
    ylabels = {
        'expect_ratio': 'Expectation ratio',
        'expect_ratio_rand': 'Expectation ratio increase',
        'default': 'Expectation ratio'
    }
    for index, ax in g.axes_dict.items():
        if row is None:
            x = index
            y = 'default'
        elif col is None:
            x = 'default'
            y = index
        else:
            x, y = index
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            if y in ylabels.keys():
                ax.set_ylabel(ylabels[y])
        if x in xlabels.keys():
            ax.set_xlabel(xlabels[x])
    legend = g.axes[0, 0].legend()
    for lh in legend.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([50])
    if show:
        plt.show()
    return g


def eval_optimization_path(path, algorithm, shots, run_id=None):
    if isinstance(path, str):
        path = Path(path)
    df = pd.read_csv(path / 'param_map/parameter_map.csv')
    column = f'{algorithm}_{str(shots)}'
    average(df, [column])
    column_avg = f'{column}_avg'
    row_numbers = df.shape[0]
    shape = (int(np.sqrt(row_numbers / 2)), 2 * int(np.sqrt(row_numbers / 2)))
    values = np.array(df[column_avg].to_list()).reshape(shape)
    plt.ioff()
    fig = contour_plot_grid(values, figsize=(10, 8))
    plt.ion()

    d_config, d_graph, d_result, d_execution_list = get_qaoa_execution(str((path / 'qaoa_execution').resolve()))
    key = f'{algorithm.replace("-", "_")}_results'
    alg_results = d_result[key]
    ax = fig.axes[0]
    if run_id is None:
        for run in alg_results:
            y = [item[0] for item in run['optimization_path']]
            x = [item[1] for item in run['optimization_path']]
            ax.plot(x, y)
    else:
        y = [item['_x'][0] % np.pi for item in alg_results]
        x = [item['_x'][1] % 2 * np.pi for item in alg_results]
        ax.plot(x, y, 'r+')
    fig.show()
