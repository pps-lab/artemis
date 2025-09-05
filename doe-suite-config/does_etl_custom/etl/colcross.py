from copy import deepcopy
import typing

import numpy as np
from doespy.etl.steps.colcross.colcross import BaseColumnCrossPlotLoader, SubplotConfig, PlotConfig, BrokenColumnCrossPlotLoader
from doespy.etl.steps.colcross.hooks import (CcpHooks, default_hooks)
from doespy.etl.steps.colcross.components import (BrokenSubplotGrid, SubplotGrid, Metric)
from matplotlib.ticker import MaxNLocator

from typing import Dict, List, Optional, Union
from dataclasses import replace

import gossip
from matplotlib import pyplot as plt
import pandas as pd


class MyCustomSubplotConfig(SubplotConfig):

    # INFO: We extend the default config and add a new attribute
    grid: bool = True

class MyCustomBrokenPlotConfig(PlotConfig):
    subplot_grid: Optional[BrokenSubplotGrid] = None

class MyCustomColumnCrossPlotLoader(BaseColumnCrossPlotLoader):

    # INFO: We provide a custom subplot config that extends the default config
    #       and override it here
    cum_subplot_config: List[MyCustomSubplotConfig]

    def setup_handlers(self):
        """:meta private:"""

        # NOTE: We can unregister function by name for a hook if needed
        # for x in gossip.get_hook(CcpHooks.SubplotPostChart).get_registrations():
        #    if x.func.__name__ == "ax_title":
        #        x.unregister()

        # NOTE: We can unregister all registered functions for a hook if needed
        # gossip.get_hook(SubplotHooks.SubplotPostChart).unregister_all()

        # install the class specific hooks
        MyCustomColumnCrossPlotLoader.blueprint().install()



@MyCustomColumnCrossPlotLoader.blueprint().register(CcpHooks.SubplotPostChart)
def apply_grid(
    ax: plt.Axes,
    df_subplot: pd.DataFrame,
    subplot_id: Dict[str, typing.Any],
    plot_config,
    subplot_config,
    loader,
):

   if subplot_config.grid:
      ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)



class MyCustomBrokenColumnCrossPlotLoader(MyCustomColumnCrossPlotLoader):
    cum_plot_config: List[MyCustomBrokenPlotConfig] = []

    def plot(self, df: pd.DataFrame) -> List:
        #print(f"Plot Config: {plot_config}")
        if df.empty:
            return

        for hook in gossip.get_all_hooks():
            hook.unregister_all()

        default_hooks.install()

        self.setup_handlers()

        gossip.trigger(CcpHooks.DataPreMetrics, df=df, loader=self)

        if self.metrics is not None:
            df = Metric.convert_metrics(self.metrics, df)

        gossip.trigger(CcpHooks.DataPreFilter, df=df, loader=self)

        df = self.data_filter.apply(cols=self.collect_cols(), df=df)

        gossip.trigger(CcpHooks.DataPreGroup, df=df, loader=self)

        figs = []

        for _i_plot, df_plot, plot_id in self.fig_foreach.for_each(df, parent_id={}):

            print(f"{plot_id=}")

            gossip.trigger(
                CcpHooks.FigPreInit, df_plot=df_plot, plot_id=plot_id, loader=self
            )

            if self.cum_plot_config is not None and len(self.cum_plot_config) > 0:
                plot_config = self.cum_plot_config[0].__class__.merge_cumulative(
                    configs=self.cum_plot_config, plot_id=plot_id
                )
            else:
                plot_config = None

            if plot_config is not None and plot_config.subplot_grid is not None:
                subplot_grid = plot_config.subplot_grid
            else:
                subplot_grid = SubplotGrid.empty()

            fig, axs = subplot_grid.init(df=df_plot, parent_id=plot_id)
            gossip.trigger(
                CcpHooks.FigPreGroup,
                fig=fig,
                axs=axs,
                df_plot=df_plot,
                plot_id=plot_id,
                plot_config=plot_config,
                loader=self,
            )

            for subplot_idx, df_subplot, subplot_id, (ax_top, ax_bottom) in subplot_grid.for_each(axs, df=df_plot, parent_id=plot_id):

                # -- Metric selection (unchanged) --
                if "$metrics$" in plot_id:
                    m = plot_id["$metrics$"]
                elif "$metrics$" in subplot_id:
                    m = subplot_id["$metrics$"]
                elif len(self.metrics) == 1:
                    m = next(iter(self.metrics.keys()))
                    plot_id["$metrics$"] = m
                metric = self.metrics[m]
                subplot_id["$metric_unit$"] = metric.unit_label

                data_id = {
                    **plot_id,
                    **subplot_id,
                    "subplot_row_idx": subplot_idx[0],
                    "subplot_col_idx": subplot_idx[1],
                }

                gossip.trigger(
                    CcpHooks.SubplotPreConfigMerge,
                    df_subplot=df_subplot,
                    subplot_id=data_id,
                    plot_config=plot_config,
                    cum_subplot_config=self.cum_subplot_config,
                    loader=self,
                )

                assert self.cum_subplot_config and len(self.cum_subplot_config) > 0

                subplot_config_top = self.cum_subplot_config[0].__class__.merge_cumulative(
                    configs=self.cum_subplot_config, data_id=data_id
                )
                subplot_config_bottom = deepcopy(subplot_config_top)

                model = data_id.get('model', '')

                # ========== CASE: NORMAL PLOT ==========
                # Determine if broken or normal
                is_broken = ax_bottom is not None

                # Always trigger pre-chart for top axis
                gossip.trigger(
                    CcpHooks.SubplotPreChart,
                    ax=ax_top,
                    df_subplot=df_subplot,
                    subplot_id=data_id,
                    plot_config=plot_config,
                    subplot_config=subplot_config_top,
                    loader=self,
                )

                if is_broken:
                    gossip.trigger(
                        CcpHooks.SubplotPreChart,
                        ax=ax_bottom,
                        df_subplot=df_subplot,
                        subplot_id=data_id,
                        plot_config=plot_config,
                        subplot_config=subplot_config_bottom,
                        loader=self,
                    )

                    # Broken axis visuals
                    d = 0.02
                    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
                    ax_top.plot((-d, +d), (-2*d, +2*d), **kwargs)
                    ax_top.plot((1 - d, 1 + d), (-2*d, +2*d), **kwargs)
                    kwargs.update(transform=ax_bottom.transAxes)
                    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

                    ax_top.spines['bottom'].set_visible(False)
                    ax_bottom.spines['top'].set_visible(False)

                # --- Y-limits ---
                # Normal case: use full range; Broken case: split
                model_ylim_map = {
                    'mnist':      ((0, 2),         (0.90, 2),      (0, 0.2)),
                    'resnet18':   ((0, 6.5),       (5, 6.5),        (0, 2)),
                    'dlrm':       ((0, 8),         (7, 8),          (0, 3.5)),
                    'mobilenet':  ((0, 240),       (200, 240),      (0, 83)),
                    'vgg':        ((0, 126),       (95, 126),       (0, 31)),
                    'diffusion':  ((0, 470),       (300, 470),      (0, 150)),
                    'gpt2':       ((0, 240),       (200, 240),      (0, 111)),
                }
                full_ylim, top_ylim, bottom_ylim = model_ylim_map.get(model, ((0, 1), (0, 1), (0, 1)))

                if is_broken:
                    ax_top.set_ylim(top_ylim)
                    ax_bottom.set_ylim(bottom_ylim)
                else:
                    ax_top.set_ylim(full_ylim)

                # --- Chart creation ---
                subplot_config_top.create_chart(
                    ax=ax_top,
                    df1=df_subplot,
                    data_id=data_id,
                    metric=metric,
                    plot_config=plot_config,
                )

                if is_broken:
                    subplot_config_bottom.create_chart(
                        ax=ax_bottom,
                        df1=df_subplot,
                        data_id=data_id,
                        metric=metric,
                        plot_config=plot_config,
                    )

                # --- Shared config cleaning ---
                    #subplot_config_bottom.yaxis = None

                    subplot_config_top.yaxis = None
                    subplot_config_top.xaxis.ticks = None
                    if is_broken: 
                        #subplot_config_bottom.yaxis = None
                        subplot_config_bottom.xaxis.ticks = None
                        subplot_config_bottom.ax_title = None

                # --- Axis tick density ---
                ax_top.yaxis.set_major_locator(MaxNLocator(nbins=2 if is_broken else 6))
                if is_broken:
                    ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=4))

                # --- Post hooks ---
                gossip.trigger(
                    CcpHooks.SubplotPostChart,
                    ax=ax_top,
                    df_subplot=df_subplot,
                    subplot_id=data_id,
                    plot_config=plot_config,
                    subplot_config=subplot_config_top,
                    loader=self,
                )

                if is_broken:
                    gossip.trigger(
                        CcpHooks.SubplotPostChart,
                        ax=ax_bottom,
                        df_subplot=df_subplot,
                        subplot_id=data_id,
                        plot_config=plot_config,
                        subplot_config=subplot_config_bottom,
                        loader=self,
                    )
                    #if 
                    # Add your own label as a text box
                    #print(f"{df_subplot['model'].iloc[0]=}")
                    # if df_subplot['model'].iloc[0] == 'mnist':
                    #     ax_top.text(
                    #         -0.5,               # x position (to the left)
                    #         -1.5,                 # y position (middle of y-axis)
                    #         "Your Label",
                    #         transform=ax_top.transAxes,
                    #         rotation=90,
                    #         va='center',
                    #         ha='center',
                    #         fontsize=12,
                    #         clip_on=False
                    #     )


            flat = []
            for row in axs:
                for ax_top, ax_bottom in row:
                    if ax_top is not None:
                        flat.append(ax_top)
                    if ax_bottom is not None:
                        flat.append(ax_bottom)
            flat = np.array(flat)
            
            gossip.trigger(
                CcpHooks.FigPost,
                fig=fig,
                axs=flat,
                df_plot=df_plot,
                plot_id=plot_id,
                plot_config=plot_config,
                loader=self,
            )


            figs.append((plot_id, df_plot, fig))

        return figs