from doespy.etl.steps.colcross.base import BaseSubplotConfig, is_match
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from enum import Enum
from doespy.design.etl_design import MyETLBaseModel

import typing
import pandas as pd
from typing import Dict, List, Literal, NamedTuple, Optional, Union

from doespy.design.etl_design import MyETLBaseModel

from pydantic import field_validator, ConfigDict, Field


class SubplotGrid(MyETLBaseModel):
    """
    Create a grid of subplots within a single figure, where each subplot corresponds to a unique combination of values from specified row and column columns.
    """

    rows: List[str]
    """The columns that are used to define the rows in the subplot grid.

    Create a row for each unique combination of values from the specified row columns.
    (see ``jp_except`` for defining exceptions, i.e., not all combinations)
    """

    cols: List[str]
    """The columns that are used to define the columns in the subplot grid.

    Create a column in each row for each unique combination of values from the specified column columns.
    (see ``jp_except`` for defining exceptions, i.e., not all combinations)
    """

    jp_except: Optional[str] = None
    """Skip certain combinations of (row, col) based on the data id.
    """

    share_x: Literal["none", "all", "row", "col"] = "none"
    """Options available in plt.subplots(..) to share the x-axis across subplots"""

    share_y: Literal["none", "all", "row", "col"] = "row"
    """Options available in plt.subplots(..) to share the y-axis across subplots"""

    class WidthHeight(NamedTuple):
        w: float
        h: float

    subplot_size: WidthHeight = WidthHeight(2.5, 2.5)
    """Size of each subplot (width, height) -> the size of the figure is determined by the number of rows and columns in the grid."""

    kwargs: Dict[str, typing.Any] = Field(default_factory=dict)
    """kwargs for the plt.subplots(..) function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)"""

    @classmethod
    def empty(cls):
        """:meta private:"""
        return SubplotGrid(rows=[], cols=[])

    def init(
        self,
        df: pd.DataFrame,
        parent_id: Dict[str, str],
        subplot_size: WidthHeight = None,
    ):
        """:meta private:"""
        # NOTE: also assumes correctly sorted df

        def dict_to_tuple(d):
            return tuple(sorted(d.items()))

        cfe = ColsForEach(cols=self.rows + self.cols, jp_except=self.jp_except)

        grid = dict()
        for i, df1, data_id in cfe.for_each(df=df, parent_id=parent_id):
            row_id = dict_to_tuple({row: data_id[row] for row in self.rows})
            col_id = dict_to_tuple({col: data_id[col] for col in self.cols})

            if row_id not in grid:
                grid[row_id] = list()

            grid[row_id].append(col_id)

        n_rows = len(grid)

        n_cols = [len(x) for x in grid.values()]
        print("cfe: ", cfe)
        assert n_cols.count(n_cols[0]) == len(
            n_cols
        ), f"The subplots do not form a grid (not all rows have the same number of columns)  {n_rows=}  {n_cols=}  (do you use the correct jp_except condition?)"
        n_cols = n_cols[0]

        print(f"Init Subplot Grid: {n_rows=} {n_cols=}")

        #print(f"\n{grid=}\n")
        if subplot_size is None:
            subplot_size = self.subplot_size

        # due to squeeze=False -> axs is always 2D array

        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(n_cols * subplot_size.w, n_rows * subplot_size.h),
            sharex=self.share_x,
            sharey=self.share_y,
            squeeze=False,
            **self.kwargs,
        )

        return fig, axs

    def get_cols(self):
        """:meta private:"""
        return self.rows + self.cols

    def for_each(self, axs, df: pd.DataFrame, parent_id: Dict[str, str]):
        """:meta private:"""

        n_rows = len(axs)
        n_cols = len(axs[0])

        subplot_idx_lst = []

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                subplot_idx_lst.append((row_idx, col_idx))

        cfe = ColsForEach(cols=self.rows + self.cols, jp_except=self.jp_except)
        for subplot_idx, (_i, df1, data_id) in zip(
            subplot_idx_lst, cfe.for_each(df=df, parent_id=parent_id)
        ):
            yield subplot_idx, df1, data_id


class Metric(MyETLBaseModel):
    """
    The metric specifies the columns in the dataframe where the data is stored, along with details such as units.
    Additionally, it enables conversion to other units as needed.

    Each subplot will be built based on exactly one metric.
    """

    value_cols: Union[str, List[str]]
    """Identify the column(s) in the dataframe containing the data values.

    The semantic of multiple columns varies based on the chart type being used.
    For instance, in a stacked bar chart, multiple columns may represent distinct
    "stacked" segments within a bar.

    To understand the semantic of these columns, consult the documentation of the specific chart type.
    """

    error_cols: Union[str, List[str]] = None
    """Identify the column(s) in the dataframe containing the error values.

    The semantic of the error values depends on the chart type being used (see ``value_cols`` above).
    For example, in a bar chart, the error values may represent the error bars for the corresponding data values.
    """

    value_multiplicator: float = 1.0
    """A multiplicator to scale the data values to e.g., convert to a desired unit.
        For instance, if the data is in milliseconds and you want to convert it to seconds, set this value to 0.001.

        new_value = old_value * value_multiplicator / value_divider
    """

    error_multiplicator: float = 1.0
    """A multiplicator to scale the error values to e.g., convert to a desired unit.

        new_error = old_error * error_multiplicator / error_divider
    """

    value_divider: float = 1.0
    """A divider to scale the data values to e.g., convert to a desired unit.
        For instance, if the data is in milliseconds and you want to convert it to seconds, set this value to 1000.

        new_value = old_value * value_multiplicator / value_divider
    """

    error_divider: float = 1.0
    """A divider to scale the error values to e.g., convert to a desired unit.

        new_error = old_error * error_multiplicator / error_divider
    """

    unit_label: str = None
    """An option to specify the unit for data values (after the optional scaling).

    This unit can be utilized when formatting labels by using the special identifier: ``$metric_unit$`` as a placeholder.
    """


    @field_validator("value_cols", "error_cols", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """:meta private:"""
        if isinstance(v, str):
            return [v]
        return v

    @classmethod
    def convert_metrics(cls, metrics: Dict[str, "Metric"], df: pd.DataFrame):
        """
        :meta private:
        Introduce a duplicate of the df for each metric and mark it with a new column $metrics$.
        -> this allows to use $metrics$ as a column to generate different plots / subplots / groups / etc.
        """

        df1 = None
        metric_cols = set()
        for m in metrics.values():
            metric_cols.update(m.value_cols)
            if m.error_cols is not None:
                metric_cols.update(m.error_cols)

        # ensure that the metric columns are numeric
        # NOTE: If a column is both a metric and a group column, then it will not remain numeric
        # TODO [nku] could add a check for this
        metric_cols = list(metric_cols)
        df[metric_cols] = df[metric_cols].apply(pd.to_numeric)

        for metric_name, metric in metrics.items():

            df_copy = df.copy()
            df_copy["$metrics$"] = metric_name

            # scale metric
            df_copy[metric.value_cols] = (
                df_copy[metric.value_cols]
                * metric.value_multiplicator
                / metric.value_divider
            )

            assert set(metric.value_cols).issubset(
                df.columns
            ), f"Metric Value Columns: Some columns not found in DataFrame. Missing: {set(metric.value_cols) - set(df.columns)}"
            if metric.error_cols is not None:
                assert set(metric.error_cols).issubset(
                    df.columns
                ), f"Metric Error Columns: Some columns not found in DataFrame. Missing: {set(metric.error_cols) - set(df.columns)}"
                df_copy[metric.error_cols] = (
                    df_copy[metric.error_cols]
                    * metric.error_multiplicator
                    / metric.error_divider
                )

            df1 = pd.concat([df1, df_copy], axis=0) if df1 is not None else df_copy

        return df1


class DataFilter(MyETLBaseModel):
    """
    Filter the DataFrame to include only the specified values from a predefined list and establish a sorting order.

    .. code-block:: yaml
       :caption: Example

        data_filter:
          allowed:
            col1: [val1, val2]          # filter out rows where col1 is not val1 or val2 + sort col1 (in that order)
            col2: [val3, val4, val5]    # filter out rows where col2 is not val3, val4, or val5 + sort col2
            # col3 is not present, so all values are allowed

    """

    allowed: Dict[str, List[str]]
    """A dictionary that maps column names to a list of allowed values.

    If a column is not present in the dictionary, then all values are allowed.

    The order of the values in the list determines the order of the values and thus also the order in the plots.
    """

    @classmethod
    def empty(cls):
        """:meta private:"""
        return DataFilter(allowed=dict())

    def apply(self, cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """:meta private:"""

        n_rows_intial = len(df)
        df_filtered = df.copy()

        cols_values = [(col, self.allowed.get(col, None)) for col in cols]

        # filter out non-relevant results
        for col, allowed in cols_values:

            # convert column to string for filtering
            try:
                df_filtered[col] = df_filtered[col].astype(str)
            except KeyError:
                raise KeyError(f"col={col} not in df.columns={df.columns}")

            all_values = df_filtered[col].unique().tolist()

            if allowed is None:
                allowed = all_values

            # filter out non-relevant results
            df_filtered = df_filtered[df_filtered[col].isin(allowed)]

            remaining_values = set(df_filtered[col].unique().tolist())

            removed_values = set(all_values) - remaining_values

            if removed_values:
                print(
                    f"Filtering {col} to {allowed}   (remaining values: {remaining_values}  |  removed values: {removed_values})"
                )

            # convert to categorical
            df_filtered[col] = pd.Categorical(
                df_filtered[col], ordered=True, categories=allowed
            )
        df_filtered.sort_values(by=cols, inplace=True)

        print(
            f"Filtered out {n_rows_intial - len(df_filtered)} rows, now there are {len(df_filtered)} remaining rows"
        )

        return df_filtered


class LabelFormatter(MyETLBaseModel):
    """
    A label formatter that allows to customize the label based on a data_id.

    .. code-block:: yaml
       :caption: Example

        label:
          template: "{system}: {workload}"
        # for data_id = {"system": "A", "workload": "B"} -> label = "A: B"

    """


    template: str
    """A template string that can contain placeholders in the form "{placeholder}".
    The placeholder corresponds to column names (which are presend in the data_id)
    """

    def apply(
        self, data_id: Dict[str, str], subplot_config: BaseSubplotConfig, info: str
    ) -> str:
        """:meta private:"""

        # template string: "Hello {name}"
        labels = {k: subplot_config.label(lbl, data_id) for k, lbl in data_id.items()}

        try:
            lbl = self.template.format(**labels)
        except KeyError as e:
            raise KeyError(
                f"LabelFormatter: {info}: Could not find all keys in data_id: {data_id}"
            ) from e
        return lbl


class KwargsLabelFormatter(LabelFormatter):

    kwargs: Dict[str, typing.Any] = Field(default_factory=dict)
    """Additional keyword arguments that will be passed to the matplotlib function that is used to set the labels.
    """


class LegendConfig(MyETLBaseModel):

    label: Union[str, LabelFormatter] = None
    """The basic label format assigned to each artist.
        Using a label in the `ArtistConfig` / `SubplotConfig` will overwrite this label.
    """

    kwargs: Dict[str, typing.Any] = Field(default_factory=dict)
    """kwargs for ax level legend (no ax level legend if None).
        e.g. {loc: "upper center", ncol: 4, bbox_to_anchor: [0.51, 0.075], columnspacing: 3.5,  fancybox: True}
    """

    def get_label(self, data_id, subplot_config):
        """:meta private:"""
        if self.label is None:
            return None
        if isinstance(self.label, str):
            return self.label
        return self.label.apply(
            data_id, subplot_config=subplot_config, info="legend"
        )


class ColsForEach(MyETLBaseModel):

    cols: List[str]
    """Performs a group by with the cols as keys and yields the resulting dataframes."""

    jp_except: Optional[str] = None
    """Skip certain combinations based on the data id (and parent data_id)"""

    label: KwargsLabelFormatter = None
    """Allows to define a label for each group based on the data_id"""

    @classmethod
    def empty(cls):
        """:meta private:"""
        return ColsForEach(cols=[])

    def get_cols(self):
        """:meta private:"""
        return self.cols.copy()

    def for_each(self, df: pd.DataFrame, parent_id: Dict[str, str]):
        """:meta private:"""

        if not self.cols:  # is empty
            yield 0, df, {}
        else:
            cols = self.cols[0] if len(self.cols) == 1 else self.cols

            i = 0
            for idx, df1 in df.groupby(cols, dropna=False):
                idx = (idx,) if not isinstance(idx, tuple) else idx
                if len(self.cols) != len(idx):
                    raise ValueError(f"Length mismatch: {len(self.cols)} vs {len(idx)}")
                data_id = {k: v for k, v in zip(self.cols, list(idx))}
                all_id = {**parent_id, **data_id}
                if self.jp_except is None or not is_match(
                    self.jp_except, all_id, info="jp_except"
                ):
                    yield i, df1, data_id
                    i += 1
                else:
                    print(f"Skipping {all_id} due to jp_except={self.jp_except}")


def round_short_axis_formatter(value, _pos):
    """
    Custom formatting function for y-axis labels.
    """

    def format(value):

        if abs(value) < 0.001:
            formatted_number = f"{value:.4f}"
        elif abs(value) < 0.01:
            formatted_number = f"{value:.3f}"
        elif abs(value) < 0.1:
            formatted_number = f"{value:.2f}"
        else:
            formatted_number = f"{value:.1f}"

        # remove trailing zero
        if "." in formatted_number:
            formatted_number = formatted_number.rstrip("0").rstrip(".")

        return formatted_number

    if abs(value) >= 1e9:

        formatted_number = format(value / 1e9)
        formatted_number += "B"
        val = formatted_number
    elif abs(value) >= 1e6:
        formatted_number = format(value / 1e6)
        formatted_number += "M"
        val = formatted_number

    elif abs(value) >= 1e3:
        formatted_number = format(value / 1e3)
        formatted_number += "k"
        val = formatted_number
    else:
        formatted_number = format(value)
        val = formatted_number

    if val == "100M":
        val = "0.1B"
    return val


axis_formatter = {
    "round_short": round_short_axis_formatter
    # NOTE: could add other formatting functions
}


AxisFormatter = Enum("AxisFormatter", [(f, f) for f in axis_formatter.keys()])


class AxisConfig(MyETLBaseModel):
    model_config = ConfigDict(use_enum_values=True)

    scale: Literal["linear", "log", "symlog", "logit"] = None
    """The scale of the axis"""

    label: LabelFormatter = None
    """Axis Label"""

    class AxisLim(MyETLBaseModel):
        min: Union[
            float, Dict[Literal["data_max_scaler", "data_min_scaler"], float]
        ] = 0.0
        max: Union[
            float, Dict[Literal["data_max_scaler", "data_min_scaler"], float]
        ] = None

        def limits(self, data_interval):

            def compute_lim(x):
                if isinstance(x, dict):
                    assert len(x) == 1, "can only have one key in min"
                    if "data_min_scaler" in x:
                        return x["data_min_scaler"] * data_interval[0]
                    elif "data_max_scaler" in x:
                        return x["data_max_scaler"] * data_interval[1]
                return x

            return compute_lim(self.min), compute_lim(self.max)

    lim: AxisLim = None
    """Define the limits of the axis. Can either define a fixed value or scale the limits based on the data interval."""

    ticks: Union[Dict, int] = (
        None
    )
    """Set axis ticks (corresponds to matplotlib function set_xticks / set_yticks or it's the number of ticks)

    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticks.html
    """


    tick_params: Union[Dict, List[Dict]] = None
    """Additional options for the tick_params function.

    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    """

    major_formatter: AxisFormatter = None
    """Label formatting function for major ticks on the axis.

    The aviailable options are:
    - "round_short" -> which rounds the number to a short format (e.g., 1.2M)
    """

    minor_formatter: AxisFormatter = None
    """Label formatting function for major ticks on the axis.

    The aviailable options are:
    - "round_short" -> which rounds the number to a short format (e.g., 1.2M)
    """


class ArtistConfig(MyETLBaseModel):
    """
    Configure settings for a specific artist (e.g., line, bar, scatter, etc.).

    This configuration allows customization beyond the predefined fields listed below.
    Additional options can be passed as keyword arguments (kwargs) to the matplotlib
    function corresponding to the artist.

    For instance, setting {color: blue} for a bar plot would define the bars' color as blue.

    Refer to the specific artist or chart type documentation for a comprehensive list of
    available customization options.
    """


    jp_query: Optional[str] = None
    """The JMESPath query is applied to the artist_id (i.e., dict of col:data pairs) to determine whether this configuration entry applies or not.
    If the jp_query matches, then this configuration applies.
    If None, then this config applies to all artist_ids."""

    label: Union[LabelFormatter, str] = None
    """Special label formatter that allows to customize the label used for the artist."""
    model_config = ConfigDict(extra="allow")

    @classmethod
    def merge_cumulative(
        cls,
        configs: List["ArtistConfig"],
        data_id: Dict,
        subplot_config: "BaseSubplotConfig",
        info="artist_config",
    ) -> Dict:
        """:meta private:"""
        config = {}

        # loop in reverse order to give the first filter the highest priority
        for cfg in reversed(configs):
            if is_match(cfg.jp_query, data_id, info):

                d = cfg.model_dump().copy()
                if hasattr(cfg, "label") and isinstance(cfg.label, LabelFormatter):
                    d["label"] = cfg.label.apply(data_id, subplot_config, info)
                config.update(d)

        if "jp_query" in config:
            del config["jp_query"]

        return config



class BrokenSubplotGrid(SubplotGrid):
    def init(
        self,
        df: pd.DataFrame,
        parent_id: Dict[str, str],
        subplot_size: SubplotGrid.WidthHeight = None,
    ):
        """:meta private:"""
        # NOTE: also assumes correctly sorted df

        def dict_to_tuple(d):
            return tuple(sorted(d.items()))
        print(f"{df['model']=}")
        cfe = ColsForEach(cols=self.rows + self.cols, jp_except=self.jp_except)

        grid = dict()
        for i, df1, data_id in cfe.for_each(df=df, parent_id=parent_id):
            row_id = dict_to_tuple({row: data_id[row] for row in self.rows})
            col_id = dict_to_tuple({col: data_id[col] for col in self.cols})

            if row_id not in grid:
                grid[row_id] = list()

            grid[row_id].append(col_id)

        n_rows = len(grid)

        n_cols = [len(x) for x in grid.values()]
        print("cfe: ", cfe)
        assert n_cols.count(n_cols[0]) == len(
            n_cols
        ), f"The subplots do not form a grid (not all rows have the same number of columns)  {n_rows=}  {n_cols=}  (do you use the correct jp_except condition?)"
        n_cols = n_cols[0]

        print(f"Init Subplot Grid: {n_rows=} {n_cols=}")

        #print(f"\n{grid=}\n")
        if subplot_size is None:
            subplot_size = self.subplot_size

        # due to squeeze=False -> axs is always 2D array
        # self.kwargs.pop("constrained_layout", None)
        # height_ratios = []
        # for i in range(n_rows):
        #     for j in range(n_cols):
        #         if i == 1 and j in [3, 6]:  # Second row, 4th and 7th plots
        #             height_ratios.append(0)  # Skip top
        #             height_ratios.append(1)  # Only use bottom
        #         else:
        #             height_ratios.append(1)
        #             height_ratios.append(2)
        self.kwargs['height_ratios'] = [(i % 2) + 1 for i in range(n_rows * 2)]
        fig, axs = plt.subplots(
            nrows=2 * n_rows,
            ncols=n_cols,
            figsize=(n_cols * subplot_size.w, n_rows * subplot_size.h),
            sharex=True,
            sharey=False,
            squeeze=False,
            #constrained_layout=False,
            **self.kwargs,
        )
        # return fig, axs

    
        # Row 2 (index 1), columns 4 and 7 (indices 3 and 6) = normal
        normal_positions = {(1, 3), (1, 6)}

        # ------------------------
        # Build the grid
        # ------------------------
        fig = plt.figure(
            figsize=(n_cols * subplot_size.w, n_rows * subplot_size.h),
            constrained_layout=True
        )

        # Two grid rows per logical row: [1, 2] per row
        height_ratios = []
        for _ in range(n_rows):
            height_ratios.extend([1, 2])

        gs = fig.add_gridspec(nrows=2 * n_rows, ncols=n_cols, height_ratios=height_ratios, hspace=0.02)

        # ------------------------
        # Create axes grid
        # ------------------------
        axs = [[None for _ in range(n_cols)] for _ in range(n_rows)]

        for row in range(n_rows):
            for col in range(n_cols):
                if (row, col) in normal_positions:
                    # Normal subplot spanning both rows
                    ax = fig.add_subplot(gs[2 * row : 2 * row + 2, col])
                    axs[row][col] = (ax, None)
                    pos = ax.get_position()
                    # ax_top.set_position([
                    #     pos.x0, 
                    #     pos.y0, 
                    #     pos.width - 0.1, 
                    #     pos.height - 0.1,
                    # ])
                else:
                    # Broken axis: top and bottom
                    ax_top = fig.add_subplot(gs[2 * row, col])
                    ax_bot = fig.add_subplot(gs[2 * row + 1, col], sharex=ax_top)
                    ax_top.tick_params(labelbottom=False)
                    axs[row][col] = (ax_top, ax_bot)
                    hspace = 0.01375
                    y0 = 0.035
                    y1 = 0.85
                    x0 = -0.1
                    width = 0.2
                    if True:

                        top_pos = ax_top.get_position()
                        bot_pos = ax_bot.get_position()

                        # Compute the vertical center between the two
                        mid_y = (top_pos.y0 + bot_pos.y1) / 2
                        gap = top_pos.y0 - bot_pos.y1

                        # Pull them slightly toward each other by reducing the gap
                        shift = gap * 0.4  # tune this from 0.2 to 0.6 as needed

                        # ax_top.set_position([
                        #     top_pos.x0, 
                        #     top_pos.y0, 
                        #     top_pos.width,
                        #     top_pos.height,
                        # ])

                        # ax_bot.set_position([
                        #     bot_pos.x0, 
                        #     bot_pos.y0, 
                        #     bot_pos.width,
                        #     bot_pos.height,
                        # ])

                        # ax_top_pos = ax_top.get_position()
                        # ax_bot_pos = ax_bot.get_position()
                        # # Shrink vertical space by adjusting bottom subplot's position
                        # ax_top.set_position([
                        #     ax_top_pos.x0 - 0.1,
                        #     ax_top_pos.y0 + 0.02,  # move it up
                        #     ax_top_pos.width + 0.02,
                        #     ax_top_pos.height -0.05
                        # ])

                        # # Optionally: tighten top subplot
                        # ax_bot.set_position([
                        #     ax_top_pos.x0 - 0.1,
                        #     ax_top_pos.y0 - 0.02,  # move it down
                        #     ax_top_pos.width + 0.02,
                        #     ax_top_pos.height - 0.05
                        # ])
        return fig, axs

    def for_each(self, axs, df: pd.DataFrame, parent_id: Dict[str, str]):
        """
        Iterates over all subplots.

        Each item in `axs` is a tuple: (top_ax, bottom_ax) or (ax, None)
        """
        n_rows = len(axs)
        n_cols = len(axs[0])

        subplot_idx_lst = [(row, col) for row in range(n_rows) for col in range(n_cols)]

        cfe = ColsForEach(cols=self.rows + self.cols, jp_except=self.jp_except)

        for subplot_idx, (_i, df1, data_id) in zip(subplot_idx_lst, cfe.for_each(df=df, parent_id=parent_id)):
            row, col = subplot_idx
            ax_tuple = axs[row][col]  # now always (ax_top, ax_bottom) or (ax, None)
            yield subplot_idx, df1, data_id, ax_tuple
