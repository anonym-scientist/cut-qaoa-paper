import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from qaoa.circuit_generation import create_qaoa_circ_parameterized


def two_heatmaps(df, column_name_1, column_name_2, shape, chart_title, titles):
    min_val = min(df[column_name_1].min(), df[column_name_2].min())
    max_val = max(df[column_name_1].max(), df[column_name_2].max())
    values_1 = np.array(df[column_name_1].to_list()).reshape(shape)
    values_2 = np.array(df[column_name_2].to_list()).reshape(shape)
    fig = plt.figure()

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 2),
                    axes_pad=0.05,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    for val, t, ax in zip([values_1, values_2], titles, grid):
        im = ax.imshow(val, extent=(0, 2 * np.pi, 0, 2 * np.pi), vmin=min_val, vmax=max_val, origin='lower')
        ax.set_title(t)
        ax.set_xlabel('gamma')
        ax.set_ylabel('beta')

    grid.cbar_axes[0].colorbar(im)
    fig.suptitle(chart_title, fontsize=20)
    plt.show()


def heatmaps(df, column_names, shape, chart_title, titles, nrows_ncols, extent=(0, 2 * np.pi, 0, 2 * np.pi),
             same_color=True, dir_path=None):
    values = []
    min_val = df[column_names[0]].min()
    max_val = df[column_names[0]].max()
    for col in column_names:
        min_val = min(min_val, df[col].min())
        max_val = max(max_val, df[col].max())
        values.append(np.array(df[col].to_list()).reshape(shape))

    if same_color:
        cbar_mode = 'single'
    else:
        cbar_mode = 'each'

    fig = plt.figure(figsize=(9, 3))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=nrows_ncols,
                    axes_pad=(0.05, 0.0),
                    share_all=True,
                    label_mode="L",
                    cbar_location="bottom",
                    cbar_mode=cbar_mode,
                    cbar_pad=0.5
                    )

    for i, (val, t, ax) in enumerate(zip(values, titles, grid)):
        if same_color:
            im = ax.imshow(val, extent=extent, vmin=min_val, vmax=max_val, origin='lower')
            grid.cbar_axes[0].colorbar(im)
        else:
            im = ax.imshow(val, extent=extent, origin='lower')
            grid.cbar_axes[i].colorbar(im)

        ax.set_title(t)
        ax.set_xlabel('gamma')
        ax.set_ylabel('beta')

    fig.suptitle(chart_title, fontsize=20)
    plt.tight_layout()
    plt.show()
    if dir_path is not None:
        fig.savefig(f'{dir_path}/{chart_title}.png')


def contour_plot(df, column_names, shape, chart_title, titles, nrows_ncols, extent=(0, 2 * np.pi, 0, np.pi),
                 dir_path=None):
    values = []
    for col in column_names:
        values.append(np.array(df[col].to_list()).reshape(shape))

    fig = plt.figure(figsize=(6, 9))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=nrows_ncols,
                    axes_pad=(0.1, 0.4),
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode='each',
                    cbar_pad=0.1
                    )

    for i, (val, t, ax) in enumerate(zip(values, titles, grid)):
        im = ax.contourf(val, extent=extent, levels=15)
        grid.cbar_axes[i].colorbar(im)
        ax.set_title(t)

    fig.suptitle(chart_title, fontsize=20)
    plt.tight_layout()
    plt.show()
    if dir_path is not None:
        fig.savefig(f'{dir_path}/{chart_title}.png')


def single_contour_plot(df, column_name, shape, title=None, v_min_max=None, dir_path=None, cmap=None, colorbar=True,
                        levels=20, figsize=(7.5, 6)):
    values = np.array(df[column_name].to_list()).reshape(shape)
    fig = contour_plot_grid(values, title, v_min_max, cmap, colorbar, levels, figsize)
    fig.tight_layout()
    if dir_path is not None:
        if title is not None:
            fig.savefig(f'{dir_path}/s_c_{title}_{column_name}.pdf')
        else:
            fig.savefig(f'{dir_path}/s_c_{column_name}.pdf')


def contour_plot_grid(grid, title=None, v_min_max=None, cmap=None, colorbar=True, levels=20, figsize=None):
    plt.rc('font', size=24)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)
    if v_min_max is None:
        cb = ax.contourf(grid, extent=(0, 2 * np.pi, 0, np.pi), levels=levels, cmap=cmap)
    else:
        cb = ax.contourf(grid, extent=(0, 2 * np.pi, 0, np.pi), levels=levels, vmin=v_min_max[0], vmax=v_min_max[1],
                         cmap=cmap)
    # remove white lines in pdf
    for c in cb.collections:
        c.set_edgecolor("face")
    if colorbar:
        cbar = fig.colorbar(cb)
        cbar.solids.set_edgecolor("face")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.set_title(title)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\beta$')
    ax.set_yticks(np.arange(0, np.pi + 0.01, np.pi / 2))
    ax.set_yticklabels(['$0$', r'$\dfrac{\pi}{2}$', r'$\pi$'])
    ax.set_xticks(np.arange(0, 2 * np.pi + 0.01, np.pi))
    ax.set_xticklabels(['$0$', r'$\pi$', r'$2\pi$'])
    plt.tight_layout()
    plt.show()
    return fig


def get_circuit_from_graph(path, **kwargs):
    graph = nx.read_adjlist(path)
    node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
    nx.relabel_nodes(graph, node_mapping, copy=False)
    circuit = create_qaoa_circ_parameterized(graph, 1)
    circuit = circuit.bind_parameters([1, 1])
    circuit.remove_final_measurements(inplace=True)
    circuit.draw(output='mpl', **kwargs)
    plt.tight_layout()
    plt.show()
