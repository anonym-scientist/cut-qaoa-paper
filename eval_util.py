import json
from ast import literal_eval
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from qaoa.expectations import maxcut_obj
from graphs import get_graph_from_file


def average(df, columns):
    """
    Average of columns containing a list of numbers
    :param df: dataframe
    :param columns:
    """
    for col in columns:
        df[f'{col}_avg'] = df[col].apply(lambda values: np.average(literal_eval(values)))


def variance(df, columns):
    """
    Variance of columns containing a list of numbers
    :param df: dataframe
    :param columns:
    """
    for col in columns:
        df[f'{col}_var'] = df[col].apply(lambda values: np.var(literal_eval(values)))


def get_random_samples_from_graph(graph, shots):
    rng = np.random.default_rng()
    rints = rng.integers(low=0, high=2 ** graph.number_of_nodes(), size=shots)
    random_sample = Counter(rints)
    random_sample = {k: v / shots for k, v in random_sample.items()}
    return random_sample


def get_maximally_mixed_obj_value(graph, shots):
    samples = get_random_samples_from_graph(graph, shots)
    format_str = '{0:0' + str(graph.number_of_nodes()) + 'b}'
    value = 0
    for state, count in samples.items():
        obj = maxcut_obj(format_str.format(state), graph)
        value += obj * count
    return value


def get_max_qaoa(df, combine_algs=None, group_columns=None):
    if combine_algs is None:
        combine_algs = ['qaoa', 'qaoa_short']

    if group_columns is None:
        group_columns = ['exp_id']

    df_filtered = df[df['algorithm'].isin(combine_algs)].copy()
    df_max = df_filtered.groupby(group_columns, group_keys=False).max()
    df_max.loc[:, 'algorithm'] = ['max_qaoa' for i in range(len(df_max))]
    return pd.concat([df, df_max])


def get_max_qaoa_all_runs(df, max_column, combine_algs=None, group_columns=None):
    if combine_algs is None:
        combine_algs = ['qaoa', 'qaoa_short']

    if group_columns is None:
        group_columns = ['exp_id']

    df_filtered = df[df['algorithm'].isin(combine_algs)].copy()
    df_sum = df_filtered.groupby(group_columns, group_keys=False).sum()
    df_variants = df_sum.loc[df_sum.groupby('exp_id')[max_column].idxmax()].reset_index()
    df_max = df_filtered.merge(df_variants, on=['exp_id', 'algorithm'], how='inner', suffixes=('', '_right'))
    df_max.drop(df_max.filter(regex='_right$').columns, axis=1, inplace=True)
    df_max.loc[:, 'algorithm'] = ['max_qaoa_all' for i in range(len(df_max))]
    return pd.concat([df, df_max])


def add_config_for_exp_id(df, exp_id, path=None):
    if path is None:
        path = Path('experiment_complete')
    path = path / str(exp_id)
    config = None
    df_props = None
    for d in path.iterdir():
        if not d.is_dir():
            continue
        if d.name.find('simulator') > -1:
            continue
        properties_path = d / 'properties.csv'
        if not properties_path.exists():
            store_config_info(d)
        df_props = pd.read_csv(properties_path.resolve(), index_col=0)
        config_path = d / 'config.json'
        with open(config_path.resolve()) as f:
            config = json.load(f)
        break
    assert config is not None
    df['graph_size'] = [2 * config['graph_size'] for _ in range(df.shape[0])]
    df['backend'] = [config['backend'] for _ in range(df.shape[0])]
    assert df_props is not None
    number_of_edges = int(df_props.loc[df_props['property'] == 'number_of_edges']['value'].values[0])
    df['number_of_edges'] = [number_of_edges for _ in range(df.shape[0])]

    return df


def store_config_info(path):
    if isinstance(path, str):
        path = Path(path)
    graph_path = path / 'graph.txt'
    config_path = path / 'config.json'

    graph = get_graph_from_file(graph_path.resolve())
    with open(config_path.resolve()) as f:
        config = json.load(f)

    props = []
    value = []

    for k, v in config.items():
        props.append(k)
        value.append(v)

    props.append('number_of_nodes')
    value.append(graph.number_of_nodes())
    props.append('number_of_edges')
    value.append(graph.number_of_edges())

    df_dict = {'property': props, 'value': value}
    df = pd.DataFrame(df_dict)
    df_path = path / 'properties.csv'
    df.to_csv(df_path.resolve())
