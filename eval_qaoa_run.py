import json
import os
import re
from collections import defaultdict
from heapq import nlargest
from pathlib import Path

import matplotlib.patheffects as path_effects
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from qiskit.providers.aer import QasmSimulator

from eval_util import get_random_samples_from_graph, get_max_qaoa, get_max_qaoa_all_runs, add_config_for_exp_id
from qaoa.circuit_generation import create_qaoa_circ_parameterized
from qaoa.expectations import maxcut_obj, compute_expectation_cut
from qaoa.objective import ObjectiveFunction
from qaoa.qaoa_functions import brutforce_max_cut, brutforce_max_cut_multi_core
from utils import get_dir


def get_objective_value_distribution(counts, graph):
    format_str = '{0:0' + str(graph.number_of_nodes()) + 'b}'
    obj_counts = defaultdict(int)
    obj_func = ObjectiveFunction()
    if isinstance(counts, dict):
        for state, count in counts.items():
            if isinstance(state, np.int64) or isinstance(state, int):
                state = format_str.format(state)
            obj = obj_func.cut_size(state, graph)
            obj_counts[obj] += count
    elif isinstance(counts, list):
        for state, prob in enumerate(counts):
            if prob != 0:
                obj = obj_func.cut_size(format_str.format(state), graph)
                obj_counts[obj] += prob
    return obj_counts


def get_average_obj_val(obj_dist):
    return sum([k * v for k, v in obj_dist.items()])


def get_distribution_chart(graph, include_random=True, normalize=True, path=None, show=True, order=None, **counts_dict):
    obj_distributions = {}
    for name, counts in counts_dict.items():
        if normalize:
            if isinstance(counts, dict):
                shots = sum(counts.values())
                counts = {k: v / shots for k, v in counts.items()}
            elif isinstance(counts, list):
                shots = sum(counts)
                counts = [v / shots for v in counts]
        obj_distributions[name] = get_objective_value_distribution(counts, graph)

    if include_random:
        rs = get_random_samples_from_graph(graph, 100000)
        obj_distributions['random'] = get_objective_value_distribution(rs, graph)

    min_val = np.infty
    max_val = -np.infty

    for distribution in obj_distributions.values():
        min_val = min(min_val, min(distribution.keys()))
        max_val = max(max_val, max(distribution.keys()))

    value_lists = defaultdict(list)
    x_labels = []
    for i in range(max_val, min_val - 1, -1):
        x_labels.append(i)
        for name, distribution in obj_distributions.items():
            value_lists[name].append(distribution[i])

    x = np.arange(len(x_labels))
    plt.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.5
    n = len(value_lists)
    bar_width = width / n
    if order is None:
        for i, (name, values) in enumerate(value_lists.items()):
            ax.bar(x - width / 2 + (i + 0.5) * bar_width, values, width=bar_width, label=name)
    else:
        skip = 0
        for i, name in enumerate(order):
            values = value_lists[name]
            if len(values) == 0:
                skip += 1
                continue
            ax.bar(x - width / 2 + (i - skip + 0.5) * bar_width, values, width=bar_width, label=name)
    plt.legend()
    ax.set_xlabel('objective value')
    ax.set_xticks(x, labels=x_labels)
    ax.set_ylabel('frequency')
    if path is not None:
        fig.savefig(path)
    if show:
        plt.show()

    _, opt_val = brutforce_max_cut(graph)
    df_exp = pd.DataFrame({'algorithm': [], 'value_type': [], 'value': []})
    for name, distribution in obj_distributions.items():
        expectation_val = get_average_obj_val(distribution)
        print(f'Expectation obj {name}:', expectation_val, expectation_val / opt_val)
        df_exp.loc[len(df_exp.index)] = [name, 'expectation', expectation_val]
        df_exp.loc[len(df_exp.index)] = [name, 'expectation_norm', expectation_val / opt_val]

    if path is not None:
        df_exp.to_csv(f'{get_dir(path)}/expectation.csv')

    # print('QAOA maxcut value', item['qaoa']['cut_val'])
    # print('cut-QAOA maxcut value', item['cut-qaoa']['cut_val'])


def get_json_dir_as_dict(path):
    results = {}
    for root, dirs, files in os.walk(path):
        dict_path = root.replace(path, '').strip('/')
        if dict_path == '':
            temp_dict = results
        else:
            try:
                temp_dict = results[dict_path]
            except KeyError:
                results[dict_path] = {}
                temp_dict = results[dict_path]
        for filename in files:
            if filename == 'graph.txt':
                graph = nx.read_adjlist(f'{root}/{filename}')
                node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
                nx.relabel_nodes(graph, node_mapping, copy=False)
                for i, j in graph.edges():
                    graph.edges[i, j]['weight'] = 1
                temp_dict['graph'] = graph
            elif filename.endswith('.json'):
                with open(f'{root}/{filename}') as f:
                    result = json.load(f)
                key = filename.split('.')[0]
                temp_dict[key] = result

    return results


def get_qaoa_execution(path):
    data = get_json_dir_as_dict(path)
    execution_list = []
    i = 0
    while True:
        try:
            execution_list.append(data[str(i)])
            i += 1
        except KeyError:
            break

    for item in execution_list:
        shots = sum(item['qaoa']['counts'].values())
        item['qaoa']['counts'] = {k: v / shots for k, v in item['qaoa']['counts'].items()}
        shots = sum(item['cut-qaoa']['counts'])
        item['cut-qaoa']['counts'] = [v / shots for v in item['cut-qaoa']['counts']]
        if 'qaoa-short' in item.keys():
            shots = sum(item['qaoa-short']['counts'].values())
            item['qaoa-short']['counts'] = {k: v / shots for k, v in item['qaoa-short']['counts'].items()}
    return data['config'], data['graph'], data.get('result'), execution_list


def get_qaoa_executions(path=None, min_exp=None, max_exp=None, config=None):
    if path is None:
        path = Path('experiment_complete')
    if not isinstance(path, Path):
        path = Path(path)
    execution_dict = {}
    for exp_dir in path.iterdir():
        if not exp_dir.is_dir() or not re.match('^\d+$', exp_dir.name):
            continue
        if min_exp is not None and int(exp_dir.name) < min_exp:
            continue
        if max_exp is not None and int(exp_dir.name) > max_exp:
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.find('simulator') > -1:
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

            qaoa_execution_path = run_dir / 'qaoa_execution'
            print(exp_dir.name)
            d_config, d_graph, d_result, d_execution_list = get_qaoa_execution(str(qaoa_execution_path.resolve()))
            execution_dict[exp_dir.name] = {
                'config': d_config,
                'graph': d_graph,
                'result': d_result,
                'execution_list': d_execution_list
            }
    return execution_dict


def aggregate_counts(execution_list, n_largest=None):
    agg_counts = {}
    for item in execution_list:
        for name, result_item in item.items():
            counts = result_item['counts']
            if n_largest is not None:
                counts = get_n_largest(counts, n_largest)
            try:
                if isinstance(counts, dict):
                    agg_counts[name] = add_dicts(agg_counts[name], counts)
                elif isinstance(counts, list):
                    agg_counts[name] = [sum(x) for x in zip(agg_counts[name], counts)]
            except KeyError:
                agg_counts[name] = counts
    return agg_counts


def add_dicts(*dicts):
    totals = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            totals[key].append(value)
    for key, values in totals.items():
        totals[key] = sum(values)
    return totals


def get_n_largest(counts, n):
    if n is None or len(counts) == n:
        return counts
    if isinstance(counts, dict):
        nlargest_keys = nlargest(n, counts, key=counts.get)
        return {key: counts[key] for key in nlargest_keys}
    elif isinstance(counts, list):
        indices = np.array(counts).argsort()[-n:][::-1]
        return [value if i in indices else 0 for i, value in enumerate(counts)]


def draw_chart(path, n_largest=None, include_random=True, normalize=True):
    config, graph, result, execution_list = get_qaoa_execution(path)
    counts = aggregate_counts(execution_list, n_largest=n_largest)
    if n_largest is None:
        figure_path = f'{path}/result_distribution.pdf'
    else:
        figure_path = f'{path}/result_distribution_{n_largest}.pdf'

    counts_dict = {}
    for key, item in counts.items():
        counts_dict[key.replace('-', '_')] = item

    get_distribution_chart(graph, include_random=include_random, normalize=normalize, path=figure_path,
                           **counts_dict)


def draw_prob_curve(path):
    config, graph, result, execution_list = get_qaoa_execution(path)
    obj_array = get_objective_value_array(graph)
    counts_sorted = execution_list_to_sorted_counts(execution_list, obj_array=obj_array)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, c in counts_sorted.items():
        ax.plot(c, label=name)
    plt.legend()
    plt.show()


def execution_list_to_sorted_counts(execution_list, aggregate=True, obj_array=None):
    if aggregate:
        counts = aggregate_counts(execution_list)
        return [sort_counts(counts, obj_array=obj_array)]
    else:
        counts_sorted = []
        for item in execution_list:
            counts_sorted.append(sort_counts({k: v['counts'] for k, v in item.items()}, obj_array=obj_array))
        return counts_sorted


def execution_list_to_obj_distribution(execution_list, graph, aggregate=True, n_largest=None):
    if aggregate:
        agg_execution = aggregate_counts(execution_list, n_largest=n_largest)
        distribution = {}
        for alg, counts in agg_execution.items():
            distribution[alg] = get_objective_value_distribution(counts, graph)
        return [distribution]
    else:
        distributions = []
        for item in execution_list:
            distributions.append(
                {k: get_objective_value_distribution(get_n_largest(v['counts'], n_largest), graph) for k, v in
                 item.items()})
        return distributions


def execution_list_with_obj_values(execution_list, aggregate=True, obj_array=None):
    if aggregate:
        counts = aggregate_counts(execution_list)
        return [sort_counts(counts, obj_array=obj_array)]
    else:
        counts_sorted = []
        for item in execution_list:
            counts_sorted.append(sort_counts({k: v['counts'] for k, v in item.items()}, obj_array=obj_array))
        return counts_sorted


def sort_counts(counts, obj_array=None):
    counts_sorted = {}
    objective_sorted = {}
    for name, c in counts.items():
        if isinstance(c, dict):
            array = np.zeros(2 ** len(next(iter(c))))
            for state, value in c.items():
                array[int(state, 2)] = value
            c = array
        if obj_array is None:
            counts_sorted[name] = np.sort(c)[::-1]
        else:
            a = np.array(list(zip(obj_array, c)))
            a[:, 0] *= -1
            a = a[np.lexsort((a[:, 1], a[:, 0]))][::-1]
            counts_sorted[name] = a[:, 1]
            objective_sorted[name] = a[:, 0]
    return counts_sorted


def get_objective_value_array(graph):
    format_str = '{0:0' + str(graph.number_of_nodes()) + 'b}'
    objective_array = np.zeros(2 ** graph.number_of_nodes())
    obj_func = ObjectiveFunction()
    for i in range(2 ** graph.number_of_nodes()):
        state = format_str.format(i)
        obj = obj_func.cut_size(state, graph)
        objective_array[i] = obj
    return objective_array


def prob_curves(path=None, min_exp=None, max_exp=None, config=None, aggregate=True):
    if path is None:
        path = Path('experiment_complete')

    execution_dict = get_qaoa_executions(path, min_exp, max_exp, config)

    df_list = []
    for key, val in execution_dict.items():
        obj_array = get_objective_value_array(val['graph'])
        sorted_counts = execution_list_to_sorted_counts(val['execution_list'], aggregate=aggregate, obj_array=obj_array)
        for counts_item in sorted_counts:
            df = pd.DataFrame(list(counts_item.items()), columns=['algorithm', 'frequency'])
            df = df.explode('frequency').reset_index(drop=True)
            df['count_index'] = df.groupby('algorithm').cumcount()
            df['exp_id'] = [key for _ in range(df.shape[0])]
            df['counts'] = df['frequency'] * 2 ** (2 * val['config']['graph_size'])
            df['graph_size'] = [2 * val['config']['graph_size'] for _ in range(df.shape[0])]
            df['backend'] = [val['config']['backend'] for _ in range(df.shape[0])]
            df_list.append(df)

    merged_df = pd.concat(df_list)
    merged_df.reset_index(inplace=True, drop=True)
    merged_df = merged_df[merged_df['graph_size'] != 14].copy()
    grid = sns.relplot(data=merged_df, x='count_index', y='frequency', hue='algorithm', col='graph_size', kind='line',
                       errorbar=('pi', 80), facet_kws={'sharex': 'col', 'sharey': 'col'})
    grid.set(xlabel='States sorted by frequency', ylabel='Frequency')
    grid.savefig((path / 'state_frequencies.pdf').resolve())
    print('fertig')


def objective_histogram(path=None, min_exp=None, max_exp=None, config=None, aggregate=True, include_random=True,
                        n_largest=None, merge_qaoa=False):
    if path is None:
        path = Path('experiment_complete')

    execution_dict = get_qaoa_executions(path, min_exp, max_exp, config)
    df_list = []
    for key, val in execution_dict.items():
        graph = val['graph']
        obj_distribution_dict = execution_list_to_obj_distribution(val['execution_list'], graph, aggregate,
                                                                   n_largest=n_largest)
        cut, opt_value = brutforce_max_cut_multi_core(graph)
        for counts_item in obj_distribution_dict:
            if include_random:
                rs = get_random_samples_from_graph(graph, 100000 if aggregate else 10000)
                counts_item['random'] = get_objective_value_distribution(rs, graph)
            df = pd.DataFrame(list(counts_item.items()), columns=['algorithm', 'obj_value'])
            df = pd.concat([df, df["obj_value"].apply(pd.Series)], axis=1).drop(columns='obj_value').melt(
                id_vars=['algorithm'], var_name='obj_value', value_name='frequency')
            df['obj_diff'] = df['obj_value'] - opt_value
            df['appr_ratio'] = df['obj_value'] / opt_value
            df['exp_id'] = [key for _ in range(df.shape[0])]
            df['graph_size'] = [2 * val['config']['graph_size'] for _ in range(df.shape[0])]
            df['backend'] = [val['config']['backend'] for _ in range(df.shape[0])]
            df_list.append(df)

    print('Created Df')
    merged_df = pd.concat(df_list)
    merged_df.reset_index(inplace=True, drop=True)
    merged_df = merged_df[merged_df['graph_size'] != 14].copy()
    hue_order = ['cut-qaoa', 'qaoa']
    if merge_qaoa:
        merged_df['algorithm'] = merged_df['algorithm'].apply(lambda name: name.replace('qaoa-short', 'qaoa'))
    else:
        hue_order.append('qaoa-short')
    if include_random:
        hue_order.append('random')
    plt.rc('font', size=16)
    grid = sns.relplot(data=merged_df, x='obj_diff', y='frequency', hue='algorithm',
                       col='graph_size', kind='line', facet_kws={'legend_out': False},
                       errorbar=('pi', 80), hue_order=hue_order)
    grid.set(xlabel='Difference to the opt. obj. value', ylabel='Frequency')
    grid.set(xlim=(0, 8), xticks=range(0, 9, 1))
    grid.savefig((path / 'qaoa_distributions.pdf').resolve())
    plt.show()
    grid2 = sns.relplot(data=merged_df, x='appr_ratio', y='frequency', hue='algorithm',
                        col='graph_size', kind='line', facet_kws={'legend_out': False},
                        errorbar=('pi', 80), hue_order=hue_order)
    grid2.set(xlabel='Approximation ratio', ylabel='Frequency')
    grid2.savefig((path / 'qaoa_distributions_approx.pdf').resolve())
    plt.show()


def get_random_max(graph, shots):
    rs = get_random_samples_from_graph(graph, shots)
    return max(rs, key=lambda key: rs[key])


def add_median_labels(ax, fmt='.2f', fontsize=12):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', fontsize=fontsize, color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def get_appr_ratio_df(path=None, min_exp=None, max_exp=None, config=None, include_random=True):
    execution_dict = get_qaoa_executions(path, min_exp, max_exp, config)
    df_list = []
    for key, val in execution_dict.items():
        print(key)
        graph = val['graph']
        format_str = '{0:0' + str(graph.number_of_nodes()) + 'b}'
        result = val['result']
        cut, opt_value = brutforce_max_cut_multi_core(graph)
        result.pop('__final_result__', None)
        result.pop('params', None)
        alg_list = []
        appr_ratio_list = []
        diff_list = []
        expect_ratio_list = []
        init_params_list = []
        n_eval_list = []
        for alg, result_list in result.items():
            for r in result_list:
                alg_list.append(alg.replace('_results', ''))
                appr_ratio_list.append(r['cut_val'] / opt_value)
                diff_list.append(r['cut_val'] - opt_value)
                expect_ratio_list.append(r['_fun'] / opt_value)
                init_params_list.append(r['initial_params'])
                n_eval_list.append(r['_nfev'])

        if include_random:
            for i in range(50):
                alg_list.append('random')
                state = get_random_max(graph, 10000)
                cut_value = maxcut_obj(format_str.format(state), graph)
                appr_ratio_list.append(cut_value / opt_value)
                diff_list.append(cut_value - opt_value)
                expect_ratio_list.append(0)
                init_params_list.append([0, 0])
                n_eval_list.append(0)

        df = pd.DataFrame.from_dict({'algorithm': alg_list, 'appr_ratio': appr_ratio_list, 'obj_diff': diff_list,
                                     'expect_ratio': expect_ratio_list, 'initial_params': init_params_list,
                                     'n_evals': n_eval_list})
        df['exp_id'] = [key for _ in range(df.shape[0])]
        # df['graph_size'] = [2 * val['config']['graph_size'] for _ in range(df.shape[0])]
        # df['backend'] = [val['config']['backend'] for _ in range(df.shape[0])]
        df = add_config_for_exp_id(df, key, path)
        df_list.append(df)
    print('Created Df')
    merged_df = pd.concat(df_list)
    merged_df.reset_index(inplace=True, drop=True)
    return merged_df


def appr_ratio_boxplots(path=None, min_exp=None, max_exp=None, config=None, include_random=True, merge_qaoa=False,
                        max_qaoa=False, max_qaoa_all=False):
    if path is None:
        path = Path('experiment_complete')

    merged_df = get_appr_ratio_df(path, min_exp, max_exp, config, include_random)
    merged_df = merged_df[merged_df['graph_size'] != 14].copy()
    order = ['cut_qaoa', 'qaoa']
    if merge_qaoa:
        merged_df['algorithm'] = merged_df['algorithm'].apply(lambda name: name.replace('qaoa_short', 'qaoa'))
    else:
        order.insert(2, 'qaoa_short')
    if include_random:
        order.append('random')
    if max_qaoa:
        merged_df['beta'] = merged_df['initial_params'].apply(lambda p: p[0])
        merged_df['gamma'] = merged_df['initial_params'].apply(lambda p: p[1])
        merged_df = get_max_qaoa(merged_df, group_columns=['exp_id', 'beta', 'gamma'])
        order.insert(len(order) - 1, 'max_qaoa')
    if max_qaoa_all:
        merged_df = get_max_qaoa_all_runs(merged_df, 'appr_ratio', group_columns=['algorithm', 'exp_id'])
        order.insert(len(order) - 1, 'max_qaoa_all')
    plt.rc('font', size=16)
    bp = sns.catplot(merged_df, x='algorithm', y='obj_diff', order=order,
                     kind='box', col='graph_size')
    bp.set(ylabel='Difference to the opt. obj. value', xlabel='Variant')
    for ax in bp.axes_dict.values():
        add_median_labels(ax)
    plt.tight_layout()
    bp.savefig((path / 'obj_diff_boxplot.pdf').resolve())
    plt.show()
    bp = sns.catplot(merged_df, x='algorithm', y='appr_ratio', order=order,
                     kind='box', col='graph_size')
    bp.set(ylabel='Approximation ratio', xlabel='Variant')
    for ax in bp.axes_dict.values():
        add_median_labels(ax)
    plt.tight_layout()
    bp.savefig((path / 'appr_ratio_boxplot.pdf').resolve())
    plt.show()


def better_cut(path=None, min_exp=None, max_exp=None, config=None):
    if path is None:
        path = Path('experiment_complete')

    execution_dict = get_qaoa_executions(path, min_exp, max_exp, config)
    df_list = []
    for key, val in execution_dict.items():
        result = val['result']
        result.pop('__final_result__', None)
        result.pop('params', None)
        cut_result_list = result.pop('cut_qaoa_results')
        alg_list = []
        factor_list = []
        diff_list = []
        for alg, result_list in result.items():
            for r_cut, r_qaoa_variant in zip(cut_result_list, result_list):
                alg_list.append(alg.replace('_results', ''))
                factor_list.append((r_cut['cut_val'] - 1) / (r_qaoa_variant['cut_val'] - 1))
                diff_list.append(r_cut['cut_val'] - r_qaoa_variant['cut_val'])

        df = pd.DataFrame.from_dict({'algorithm': alg_list, 'factor': factor_list, 'diff': diff_list})
        df['exp_id'] = [key for _ in range(df.shape[0])]
        df['graph_size'] = [2 * val['config']['graph_size'] for _ in range(df.shape[0])]
        df['backend'] = [val['config']['backend'] for _ in range(df.shape[0])]

        df_list.append(df)
    print('Created Df')
    merged_df = pd.concat(df_list)
    merged_df.reset_index(inplace=True, drop=True)
    merged_df = merged_df[merged_df['graph_size'] != 14].copy()
    return merged_df


def eval_parameters(path=None, min_exp=None, max_exp=None, config=None):
    if path is None:
        path = Path('experiment_complete')

    execution_dict = get_qaoa_executions(path, min_exp, max_exp, config)

    df_list = []

    for key, val in execution_dict.items():
        result = val['result']
        graph = val['graph']
        cut, opt_value = brutforce_max_cut_multi_core(graph)
        result.pop('__final_result__', None)
        result.pop('params', None)
        alg_list = []
        sim_list = []
        qpu_list = []
        init_params_list = []
        for alg, result_list in result.items():
            obj_sim_list = compute_obj_value(graph, (item['_x'] for item in result_list))
            obj_list = [item['_fun'] for item in result_list]
            params = [item['initial_params'] for item in result_list]
            for sim, qpu, p in zip(obj_sim_list, obj_list, params):
                alg_list.append(alg.replace('_results', ''))
                sim_list.append(sim / opt_value)
                qpu_list.append(qpu / opt_value)
                init_params_list.append(p)
        df = pd.DataFrame.from_dict({'algorithm': alg_list, 'sim_obj_value': sim_list, 'qpu_obj_value': qpu_list,
                                     'initial_params': init_params_list})
        df['exp_id'] = [key for _ in range(df.shape[0])]
        df['graph_size'] = [2 * val['config']['graph_size'] for _ in range(df.shape[0])]
        df['backend'] = [val['config']['backend'] for _ in range(df.shape[0])]
        df_list.append(df)
    merged_df = pd.concat(df_list)
    merged_df.reset_index(inplace=True, drop=True)
    return merged_df


def compute_obj_value(graph, parameters):
    circuit = create_qaoa_circ_parameterized(graph, 1)
    circuit.remove_final_measurements()
    circuit.save_statevector()
    circuits_with_params = [circuit.bind_parameters(params) for params in parameters]
    backend = QasmSimulator(method='statevector')
    job = backend.run(circuits_with_params)
    res = job.result()
    probs = []
    for c in circuits_with_params:
        probs.append(res.get_statevector(c).probabilities())
    obj_func = ObjectiveFunction()
    # reuse compute expectation from cut since it already handles probability arrays
    return [compute_expectation_cut(p, graph, obj_func=obj_func.cut_size, quiet=True) for p in probs]
