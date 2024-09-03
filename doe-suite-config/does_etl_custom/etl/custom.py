from doespy.etl.steps.extractors import Extractor
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader

import pandas as pd
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import os
import json
import statistics
import ast
import yaml

class ErrorExpectedFileExtractor(Extractor):

    expected_file: List[str]

    file_regex: Union[str, List[str]] = ["^stderr.log$"]

    def extract(self, path: str, options: Dict) -> List[Dict]:

        dir = os.path.dirname(path)
        errors = []

        for x in self.expected_file:
            expected_file = os.path.join(dir, x)
            if not os.path.exists(expected_file):
                errors.append({"file": path, "error": f"missing expected file: {x}"})
        return errors




class OsirisPreprocessTransformer(Transformer):


#    col: str
#    """Name of condition column in data frame."""
#
#    dest: str
#    """Name of destination column in data frame."""
#
#    value: Dict[Any, Any]
#    """Dictionary of replacement rules:
#        The dict key is the entry in the condition ``col`` and
#        the value is the replacement used in the ``dest`` column."""


    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        # parse prover time
        df['prover_time_sec'] = (df['Prover time'].replace({'s':'*1'}, regex=True)
                                            .dropna()
                                            .apply(pd.eval))



        def aggregate(row):


            measurements = yaml.safe_load(row['Verifier time'])

            # NOTE: The conversion parsing the string to float is not very robust
            measurements_ms = []

            for m in measurements:
                if m.endswith('ms'):
                    m = float(m.replace('ms', ''))
                elif m.endswith('s'):
                    m = float(m.replace('s', '')) * 1000
                measurements_ms.append(m)

            row['mean(verifier_time_ms)'] = statistics.mean(measurements_ms)
            row['count(verifier_time_ms)'] = len(measurements_ms)
            row['min(verifier_time_ms)'] = min(measurements_ms)
            row['max(verifier_time_ms)'] = max(measurements_ms)
            row['stddev(verifier_time_ms)'] = statistics.stdev(measurements_ms)

            return row

        # parse and aggregate verifier time measurements
        df = df.apply(aggregate, axis=1)

        return df
