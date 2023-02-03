import json
import logging
import os
import sys
from ast import literal_eval
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from charts import contour_plot, single_contour_plot
from circuit_cutting import preprocess, util
from circuit_cutting.execute import ExecutorResults
from circuit_cutting.postprocess import QaoaPostProcessor
from eval_util import get_maximally_mixed_obj_value, average
from graphs import get_graph_from_file
from json_utils import store_kwargs_as_json
from qaoa.circuit_generation import create_qaoa_circ_parameterized
from qaoa.expectations import compute_expectation, _theta_to_postprocess_params, compute_expectation_cut
from qaoa.objective import ObjectiveFunction
from utils import get_dir

logger = logging.getLogger(__name__)


def load_data_from_executor(path, shot_list, mitigators=None):
    exec_results = ExecutorResults.load_results(f'{path}/executor_results.json')
    with open(f'{path}/config.json') as f:
        config = json.load(f)

    parameters = config['parameters']
    evaluations = config.get('evaluations', 1)
    reduced = config['reduced']
    short_circuits = config.get('short_circuits', False)
    cuts = config.get('n_links', 2)
    cut_shot_factor = config['cut_shot_factor']

    graph = nx.read_adjlist(f'{path}/graph.txt')
    node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
    nx.relabel_nodes(graph, node_mapping, copy=False)
    circuit = create_qaoa_circ_parameterized(graph, 1)
    sub_graph_size = graph.number_of_nodes() // 2
    sub_circuits, sub_circuits_info = preprocess.split(circuit, [sub_graph_size, sub_graph_size], reduced)
    keys, sub_circuit_list = util.dict_to_lists(sub_circuits)
    for i, j in graph.edges():
        graph.edges[i, j]['weight'] = 1

    df_dict = {'parameters': parameters}

    for shots in shot_list:
        result = defaultdict(list)
        logger.info(f'Shots: {shots}')
        if reduced:
            shots_cut = cut_shot_factor * int(shots / (2 * (4 ** cuts)))
        else:
            shots_cut = cut_shot_factor * int(shots / (2 * (5 ** cuts)))

        for i, params in enumerate(parameters):
            qaoa_result = {'params': params, 'evaluations': []}
            cut_qaoa_result = {'params': params, 'evaluations': []}
            for j in range(evaluations):
                counts = exec_results.get_counts_by_name_from_memory(str((params, j)), shots=shots)
                qaoa_result['evaluations'].append(counts)
                counts_cut = exec_results.get_counts_by_name_from_memory(str((params, j)) + '#cut', shots=shots_cut)
                counts_cut_dict = util.lists_to_dict(keys, counts_cut)
                counts_cut_dict_json = {}
                for fragment_id, fragment_dict in counts_cut_dict.items():
                    fragment_dict_json = {}
                    for key, val in fragment_dict.items():
                        fragment_dict_json[str(key)] = {'key': key, 'counts': val}
                    counts_cut_dict_json[fragment_id] = fragment_dict_json
                cut_qaoa_result['evaluations'].append(counts_cut_dict_json)
            result['qaoa'].append(qaoa_result)
            result['cut-qaoa'].append(cut_qaoa_result)

            if mitigators is not None:
                for mit in mitigators:
                    qaoa_result = {'params': params, 'evaluations': []}
                    cut_qaoa_result = {'params': params, 'evaluations': []}
                    for j in range(evaluations):
                        counts = exec_results.get_counts_by_name_from_memory(str((params, j)), shots=shots,
                                                                             mitigator=mit)
                        qaoa_result['evaluations'].append(counts)
                        counts_cut = exec_results.get_counts_by_name_from_memory(str((params, j)) + '#cut',
                                                                                 shots=shots_cut, mitigator=mit)
                        counts_cut_dict = util.lists_to_dict(keys, counts_cut)
                        counts_cut_dict_json = {}
                        for fragment_id, fragment_dict in counts_cut_dict.items():
                            fragment_dict_json = {}
                            for key, val in fragment_dict.items():
                                fragment_dict_json[str(key)] = {'key': key, 'counts': val}
                            counts_cut_dict_json[fragment_id] = fragment_dict_json
                        cut_qaoa_result['evaluations'].append(counts_cut_dict_json)
                    result[f'qaoa_{mit}'].append(qaoa_result)
                    result[f'cut-qaoa_{mit}'].append(cut_qaoa_result)

        if short_circuits:
            for i, params in enumerate(parameters):
                qaoa_short_result = {'params': params, 'evaluations': []}
                for j in range(evaluations):
                    counts_short = exec_results.get_counts_by_name_from_memory(str((params, j)) + '#short',
                                                                               shots=shots)
                    qaoa_short_result['evaluations'].append(counts_short)
                result['qaoa-short'].append(qaoa_short_result)

        logger.info(f'Data loaded')

        qaoa_result = result['qaoa']
        cut_qaoa_result = result['cut-qaoa']

        parameters = []
        qaoa_obj_values = []
        cut_qaoa_obj_values = []

        obj_func = ObjectiveFunction()

        for r_qaoa, r_cut_qaoa in zip(qaoa_result, cut_qaoa_result):
            params_qaoa = r_qaoa['params']
            evals_qaoa = r_qaoa['evaluations']
            logger.info(params_qaoa)
            qaoa_val = []
            for e in evals_qaoa:
                counts = e[0]
                qaoa_val.append(compute_expectation(counts, graph, obj_func=obj_func.cut_size))
            qaoa_obj_values.append(qaoa_val)
            params = r_cut_qaoa['params']
            evals = r_cut_qaoa['evaluations']
            assert params_qaoa == params
            parameters.append(params)
            cut_qaoa_val = []
            for e in evals:
                counts_cut = to_fragment_dict(e)
                postprocessor = QaoaPostProcessor(counts_cut, sub_circuits_info, reduced_substitution=reduced)
                params_postprocessor = _theta_to_postprocess_params(params, postprocessor.num_cuts)
                r = postprocessor.postprocess(params_postprocessor)
                cut_qaoa_val.append(compute_expectation_cut(r, graph, obj_func=obj_func.cut_size))
            cut_qaoa_obj_values.append(cut_qaoa_val)

        df_dict[f'qaoa_{shots}'] = qaoa_obj_values
        df_dict[f'cut-qaoa_{shots}'] = cut_qaoa_obj_values
        if short_circuits:
            qaoa_short_result = result['qaoa-short']
            qaoa_short_obj_values = []
            for r_qaoa_short in qaoa_short_result:
                params_qaoa_short = r_qaoa_short['params']
                evals_qaoa_short = r_qaoa_short['evaluations']
                logger.info(params_qaoa_short)
                qaoa_short_val = []
                for e in evals_qaoa_short:
                    counts = e[0]
                    qaoa_short_val.append(compute_expectation(counts, graph, obj_func=obj_func.cut_size))
                qaoa_short_obj_values.append(qaoa_short_val)

            df_dict[f'qaoa-short_{shots}'] = qaoa_short_obj_values

        if mitigators is not None:
            for mit in mitigators:
                qaoa_result = result[f'qaoa_{mit}']
                cut_qaoa_result = result[f'cut-qaoa_{mit}']

                parameters = []
                qaoa_obj_values = []
                cut_qaoa_obj_values = []

                obj_func = ObjectiveFunction()

                for r_qaoa, r_cut_qaoa in zip(qaoa_result, cut_qaoa_result):
                    params_qaoa = r_qaoa['params']
                    evals_qaoa = r_qaoa['evaluations']
                    logger.info(params_qaoa)
                    qaoa_val = []
                    for e in evals_qaoa:
                        counts = e[0]
                        qaoa_val.append(compute_expectation(counts, graph, obj_func=obj_func.cut_size))
                    qaoa_obj_values.append(qaoa_val)
                    params = r_cut_qaoa['params']
                    evals = r_cut_qaoa['evaluations']
                    assert params_qaoa == params
                    parameters.append(params)
                    cut_qaoa_val = []
                    for e in evals:
                        counts_cut = to_fragment_dict(e)
                        postprocessor = QaoaPostProcessor(counts_cut, sub_circuits_info, reduced_substitution=reduced)
                        params_postprocessor = _theta_to_postprocess_params(params, postprocessor.num_cuts)
                        r = postprocessor.postprocess(params_postprocessor)
                        cut_qaoa_val.append(compute_expectation_cut(r, graph, obj_func=obj_func.cut_size))
                    cut_qaoa_obj_values.append(cut_qaoa_val)

                df_dict[f'qaoa_{mit}_{shots}'] = qaoa_obj_values
                df_dict[f'cut-qaoa_{mit}_{shots}'] = cut_qaoa_obj_values

    df = pd.DataFrame(df_dict)
    csv_path = f'{path}/parameter_map.csv'
    df.to_csv(csv_path)
    return csv_path




def load_data(path, reduced=True):
    with open(f'{path}/result.json') as f:
        result = json.load(f)

    graph = nx.read_adjlist(f'{path}/graph.txt')
    node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
    nx.relabel_nodes(graph, node_mapping, copy=False)
    circuit = create_qaoa_circ_parameterized(graph, 1)
    sub_graph_size = graph.number_of_nodes() // 2
    sub_circuits, sub_circuits_info = preprocess.split(circuit, [sub_graph_size, sub_graph_size], reduced)
    # TODO store sub circuit info and load it
    for i, j in graph.edges():
        graph.edges[i, j]['weight'] = 1

    qaoa_result = result['qaoa']
    cut_qaoa_result = result['cut-qaoa']

    parameters = []
    qaoa_obj_values = []
    cut_qaoa_obj_values = []

    obj_func = ObjectiveFunction()

    for r_qaoa, r_cut_qaoa in zip(qaoa_result, cut_qaoa_result):
        params_qaoa = r_qaoa['params']
        evals_qaoa = r_qaoa['evaluations']
        logger.info(params_qaoa)
        qaoa_val = []
        for e in evals_qaoa:
            counts = e[0]
            qaoa_val.append(compute_expectation(counts, graph, obj_func=obj_func.cut_size))
        qaoa_obj_values.append(qaoa_val)
        params = r_cut_qaoa['params']
        evals = r_cut_qaoa['evaluations']
        assert params_qaoa == params
        parameters.append(params)
        cut_qaoa_val = []
        for e in evals:
            counts_cut = to_fragment_dict(e)
            postprocessor = QaoaPostProcessor(counts_cut, sub_circuits_info, reduced_substitution=reduced)
            params_postprocessor = _theta_to_postprocess_params(params, postprocessor.num_cuts)
            result = postprocessor.postprocess(params_postprocessor)
            cut_qaoa_val.append(compute_expectation_cut(result, graph, obj_func=obj_func.cut_size))

        cut_qaoa_obj_values.append(cut_qaoa_val)

    df = pd.DataFrame(
        {'parameters': parameters,
         'qaoa': qaoa_obj_values,
         'cut-qaoa': cut_qaoa_obj_values})
    csv_path = f'{path}/parameter_map.csv'
    df.to_csv(csv_path)
    return csv_path


def to_fragment_dict(cut_qaoa_evaluation):
    counts_cut = {}
    for k, v in cut_qaoa_evaluation.items():
        counts_dict = {}
        for counts in v.values():
            counts_dict[tuple(counts['key'])] = counts['counts']
        counts_cut[int(k)] = counts_dict
    return counts_cut


def contour_plot_from_csv(csv_path, title='Contour Plot'):
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    average(df, columns)
    columns_avg = [f'{col}_avg' for col in columns]
    row_numbers = df.shape[0]
    shape = (int(np.sqrt(row_numbers / 2)), 2 * int(np.sqrt(row_numbers / 2)))
    contour_plot(df, columns_avg, shape, title, columns, (len(columns_avg), 1), dir_path=os.path.dirname(csv_path))


def get_v_min_max(csv_path, column):
    df = pd.read_csv(csv_path)
    average(df, [column])
    column_avg = f'{column}_avg'
    values = np.array(df[column_avg].to_list())
    return [np.min(values), np.max(values)]


def single_contour_plot_from_csv(csv_path, column, title=None, v_min_max=None, cmap=None, colorbar=True, levels=20):
    df = pd.read_csv(csv_path)
    average(df, [column])
    column_avg = f'{column}_avg'
    row_numbers = df.shape[0]
    shape = (int(np.sqrt(row_numbers / 2)), 2 * int(np.sqrt(row_numbers / 2)))
    single_contour_plot(df, column_avg, shape, title, dir_path=get_dir(csv_path), v_min_max=v_min_max, cmap=cmap,
                        colorbar=colorbar, levels=levels)


def normalize_csv(csv_path):
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    for column in columns:
        df[column] = df[column].apply(lambda values: literal_eval(values)[0])
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min()) - 1
        df[column] = df[column].apply(lambda values: [values])
    csv_path_norm = csv_path.split('.')[0] + '_norm.csv'
    df.to_csv(csv_path_norm, index=False)
    return csv_path_norm



def error(csv_path_expected, csv_path, column_expected='qaoa', file_name='errors'):
    df_expected = pd.read_csv(csv_path_expected)
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    df_expected[column_expected] = df_expected[column_expected].apply(lambda values: literal_eval(values)[0])
    error_dict = {'expected_path': csv_path_expected, 'mean_squared_error': {}, 'mean_absolute_error': {}}
    for column in columns:
        df[column] = df[column].apply(lambda values: literal_eval(values)[0])
        mse = ((df[column] - df_expected[column_expected]) ** 2).sum() / df.shape[0]
        me = (df[column] - df_expected[column_expected]).abs().sum() / df.shape[0]
        print(f'{column} mse: {mse}')
        print(f'{column} me: {me}')
        error_dict['mean_squared_error'][column] = mse
        error_dict['mean_absolute_error'][column] = me

    path = get_dir(csv_path)
    store_kwargs_as_json(path, file_name, **error_dict)


def error_all(path_expected, path, column_expected='qaoa'):
    print('Absolute:')
    error(f'{path_expected}/parameter_map.csv', f'{path}/parameter_map.csv', column_expected=column_expected,
          file_name='errors_abs')
    print('Normalized:')
    error(f'{path_expected}/parameter_map_norm.csv', f'{path}/parameter_map_norm.csv', column_expected=column_expected,
          file_name='errors_norm')


def error_plot(csv_path_expected, csv_path, column_expected='qaoa', file_name='errors'):
    df_expected = pd.read_csv(csv_path_expected)
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    df_expected[column_expected] = df_expected[column_expected].apply(lambda values: literal_eval(values)[0])
    error_dict = {'expected_path': csv_path_expected, 'mean_squared_error': {}, 'mean_absolute_error': {}}
    alg_list = []
    shot_list = []
    mse_list = []
    me_list = []
    mse_norm_list = []
    me_norm_list = []
    normalization = abs(df_expected[column_expected].max() - df_expected[column_expected].min())
    for column in columns:
        alg, shots = column.rsplit('_', 1)
        alg_list.append(alg)
        shot_list.append(int(shots))
        df[column] = df[column].apply(lambda values: literal_eval(values)[0])
        mse = ((df[column] - df_expected[column_expected]) ** 2).sum() / df.shape[0]
        me = (df[column] - df_expected[column_expected]).abs().sum() / df.shape[0]
        mse_list.append(mse)
        me_list.append(me)
        mse_norm = (((df[column] - df_expected[column_expected]) / normalization) ** 2).sum() / df.shape[0]
        mse_norm_list.append(mse_norm)
        me_norm_list.append(me / normalization)

    df = pd.DataFrame(
        {'algorithm': alg_list, 'shots': shot_list, 'mean_squared_error': mse_list, 'mean_absolute_error': me_list,
         'mean_squared_error_norm': mse_norm_list, 'mean_absolute_error_norm': me_norm_list})

    path = get_dir(csv_path)

    plt.rc('font', size=15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.set_index('shots').groupby('algorithm')['mean_absolute_error'].plot(ax=ax, x='shots', marker='o', legend=True,
                                                                           title='mean_absolute_error')
    fig.savefig(f'{path}/{file_name}_me.pdf')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.set_index('shots').groupby('algorithm')['mean_squared_error'].plot(ax=ax, x='shots', marker='o', legend=True,
                                                                          title='mean_squared_error')
    fig.savefig(f'{path}/{file_name}_mse.pdf')
    plt.show()
    df.to_csv(f'{path}/errors.csv')


def max_mixed_diff_plot(csv_path_expected, path, column_expected='qaoa', df_name='parameter_map.csv',
                        graph_name='graph.txt'):
    df_expected = pd.read_csv(csv_path_expected)
    csv_path = f'{path}/{df_name}'
    df = pd.read_csv(csv_path)
    graph = get_graph_from_file(f'{path}/{graph_name}')
    columns = list(df.columns)
    columns.pop(0)  # Remove index
    columns.pop(0)  # Remove parameters
    df_expected[column_expected] = df_expected[column_expected].apply(lambda values: literal_eval(values)[0])

    alg_list = []
    shot_list = []
    max_mixed_diff_list = []

    max_mixed_obj_value = get_maximally_mixed_obj_value(graph, 100000)

    sim_diff = (df_expected[column_expected] - max_mixed_obj_value).abs().sum() / df_expected.shape[0]

    for column in columns:
        alg, shots = column.rsplit('_', 1)
        alg_list.append(alg)
        shot_list.append(int(shots))
        df[column] = df[column].apply(lambda values: literal_eval(values)[0])
        diff = (df[column] - max_mixed_obj_value).abs().sum() / df.shape[0]
        max_mixed_diff_list.append(diff)

    df = pd.DataFrame(
        {'algorithm': alg_list, 'shots': shot_list, 'max_mixed_diff': max_mixed_diff_list})

    plt.rc('font', size=15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.set_index('shots').groupby('algorithm')['max_mixed_diff'].plot(ax=ax, x='shots', marker='o', legend=True,
                                                                      title='max_mixed_diff')
    plt.axhline(y=sim_diff, color='red', linestyle='--', label='simulator')
    plt.legend()
    fig.savefig(f'{path}/max_mixed_diff.pdf')
    plt.show()
    df.loc[len(df.index)] = ['simulator', '-', sim_diff]
    df['max_mixed_diff_norm_sim'] = df['max_mixed_diff'] / sim_diff
    df.to_csv(f'{path}/max_mixed_diff.csv')


def error_plot_all(path_expected, path, column_expected='qaoa'):
    error_plot(f'{path_expected}/parameter_map_norm.csv',
               f'{path}/parameter_map_norm.csv', column_expected=column_expected,
               file_name='errors_norm')
    error_plot(f'{path_expected}/parameter_map.csv',
               f'{path}/parameter_map.csv', column_expected=column_expected,
               file_name='errors_abs')


def all_errors(path_expected, path, column_expected='qaoa', cmap='Spectral', levels=50, shots=10000):
    error_plot(path_expected, path, column_expected)
    v_min_max = get_v_min_max(path_expected, column_expected)
    single_contour_plot_from_csv(path_expected, column_expected, v_min_max=v_min_max, cmap=cmap, levels=levels)
    single_contour_plot_from_csv(path, f'qaoa_{shots}', v_min_max=v_min_max, cmap=cmap, levels=levels)
    try:
        single_contour_plot_from_csv(path, f'qaoa-short_{shots}', v_min_max=v_min_max, cmap=cmap, levels=levels)
    except KeyError:
        pass
    single_contour_plot_from_csv(path, f'cut-qaoa_{shots}', v_min_max=v_min_max, cmap=cmap, levels=levels)


def main():
    # tuple_list = [(25, 1), (50, 1), (100, 1), (250, 1), (500, 1), (25, 2), (50, 2), (100, 2), (250, 2),  (500, 2)]
    # path = iterate_and_branch('experiment_param_map/ibmq_qasm_simulator_1648050333845944000',
    #                           iterations_branching_tuples=tuple_list)

    # contour_plot_from_csv('experiment_param_map/ibmq_qasm_simulator_1648050333845944000/parameter_map_iterations_branching.csv', title='cut-search-QAOA')
    dirs = ['experiment_param_map/18/ibmq_montreal_1659637215590594310']
    for d in dirs:
        path = load_data_from_executor(d, range(250, 750, 250))
        # path = load_data(d)
        # contour_plot_from_csv(path, title='contour')
        path = normalize_csv(path)
        # contour_plot_from_csv(path, title='contour_norm')

    # single_contour_plot_from_csv('experiment_param_map/1/ibmq_ehningen_1652972852336180188/parameter_map.csv', 'qaoa_10000')
    # error_plot('experiment_param_map/3/ibmq_qasm_simulator_1653033568247291489/parameter_map_norm.csv',
    #            'experiment_param_map/3/ibmq_ehningen_1653033589433018480/parameter_map_norm.csv')


def activate_logging():
    logger.setLevel(logging.INFO)
    logging.getLogger('circuit_cutting').setLevel(logging.DEBUG)
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(s_handler)
    logging.getLogger('circuit_cutting').addHandler(s_handler)


if __name__ == '__main__':
    activate_logging()
    # histogram_intersection_from_executor('experiment_param_map/8/ibmq_qasm_simulator_1653294303715248759',
    #                                      'experiment_param_map/8/ibmq_kolkata_1653122660842823707', [10000])
    main()
