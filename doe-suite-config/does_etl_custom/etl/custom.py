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
import warnings
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


        df["proof_size_bytes"] = df["Proof size"].apply(pd.to_numeric)



        def aggregate(row):


            measurements = yaml.safe_load(row['Verifier time'])

            # NOTE: The conversion parsing the string to float is not very robust
            measurements_ms = []

            for m in measurements:
                if m.endswith('ms'):
                    m = float(m.replace('ms', '')) / 1000
                elif m.endswith('s'):
                    m = float(m.replace('s', ''))
                measurements_ms.append(m)

            row['mean(verifier_time_sec)'] = statistics.mean(measurements_ms)
            row['count(verifier_time_sec)'] = len(measurements_ms)
            row['min(verifier_time_sec)'] = min(measurements_ms)
            row['max(verifier_time_sec)'] = max(measurements_ms)
            row['stddev(verifier_time_sec)'] = statistics.stdev(measurements_ms)

            return row

        # parse and aggregate verifier time measurements
        df = df.apply(aggregate, axis=1)


        #fake = []
        #for model in ['gpt2']:
        #    for pc_type in ['ipa', 'kzg']:
        #        fake.append({"model": model,  "cpsnark": "no_com", "pc_type": pc_type, "prover_time_sec": 0.0, "mean(verifier_time_sec)": 0.0, 'stddev(verifier_time_sec)': 0.0})
        #df_fake = pd.DataFrame(fake)
#
        #df = pd.concat([df, df_fake], ignore_index=True)


        # select relevant columns
        df = df.filter(items=['suite_name', 'suite_id', 'exp_name', 'run', 'host_type', 'model', 'cpsnark', 'pc_type', 'prover_time_sec', 'proof_size_bytes', 'mean(verifier_time_sec)', 'stddev(verifier_time_sec)'])


        # compute factor vs no_com baseline
        for data_col in ["prover_time_sec", "mean(verifier_time_sec)", "proof_size_bytes"]:
            result_col = f"{data_col}_factor_vs_nocom (no_com=1)"
            df = compute_factor_vs_baseline(df, data_col=data_col, baseline_col="cpsnark", baseline_value="no_com", result_col=result_col, group_cols=["model", "pc_type"])

            result_col = f"{data_col}_factor_vs_kzg (kzg=1)"
            df = compute_factor_vs_baseline(df, data_col=data_col, baseline_col="pc_type", baseline_value="kzg", result_col=result_col, group_cols=["model", "cpsnark"])


        return df

def compute_factor_vs_baseline(df, data_col, baseline_col, baseline_value, result_col, group_cols):


    def add_group_info(group):

        row = group[group[baseline_col] == baseline_value]
        assert len(row) == 1, f"len(row)={len(row)}"
        row = row.iloc[0]

        group[result_col] = group[data_col] / row[data_col]

        # display(group[group_cols + [baseline_col, data_col, result_col]])
        return group


    df_grouped = df.groupby(group_cols).apply(add_group_info).reset_index(drop=True)

    return df_grouped

class ResultFilterTransformer(Transformer):


    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        msg = """In the initial set of results for ipa poly on the smaller models (mnist, resnet18, dlrm, mobilenet, vgg, diffusion) the suite_name was not set correctly
        We have used a naive verification approach leading to a slow verification time.
        We filter out these results because we re-run the experiments in the suite poly-ipa with the a better verficication approach.
        """
        warnings.warn(msg)
        condition = (df["model"].isin(["mnist", "resnet18", "dlrm", "mobilenet", "vgg", "diffusion"])) & (df["cpsnark"] == "poly") & (df["pc_type"] == "ipa") & (df["suite_name"] != "poly-ipa")
        df1 = df[~condition]

        return df1



class OsirisFactorLoader(Loader):


    skip_empty: bool = False
    """Ignore empty df, if set to ``False``, raises an error if the data frame is empty."""


    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:

        output_dir = self.get_output_dir(etl_info)

        print(df.columns)

        agg_d = {"prover_time_sec_factor_vs_nocom (no_com=1)": ["min", "max"], "mean(verifier_time_sec)_factor_vs_nocom (no_com=1)": ["min", "max"], "proof_size_bytes_factor_vs_nocom (no_com=1)": ["min", "max"]}


        res1 = df.groupby(["pc_type", "cpsnark"]).agg(agg_d)
        res1.to_html(f"{output_dir}/factor_vs_nocom.html")


        res2 = df.groupby(["pc_type", "cpsnark"]).agg(agg_d)
        res2.to_html(f"{output_dir}/factor_vs_kzg.html")


class ProofSizeTableLoader(Loader):

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:

        output_dir = self.get_output_dir(etl_info)

        df1 = df[["model", "cpsnark", "pc_type", "proof_size_bytes"]]

        replace_dict = {
            'mnist': 'MNIST',
            'resnet18': 'ResNet-18',
            'dlrm': 'DLRM',
            'mobilenet': 'MobileNet',
            'vgg': 'VGG',
            'diffusion': 'Diffusion',
            'gpt2': 'GPT-2'
        }

        df1['model'] = df1['model'].replace(replace_dict)

        custom_order = ['mnist', 'resnet18', 'dlrm', 'mobilenet', 'vgg', 'diffusion', 'gpt2']
        custom_order = [replace_dict[x] for x in custom_order]

        # Convert the 'model' column to a categorical type with the specified order
        df1['model'] = pd.Categorical(df1['model'], categories=custom_order, ordered=True)

        replace_dict = {
            'no_com': 'No Com',
            'poly': 'Artemis',
            'cp_link+': 'Apollo',
            'cp_link': 'Lunar',
            'poseidon': 'Poseidon'
        }

        df1['cpsnark'] = df1['cpsnark'].replace(replace_dict)

        custom_order = ['no_com', 'poly', 'cp_link+', 'cp_link', 'poseidon']
        custom_order = [replace_dict[x] for x in custom_order]
        df1['cpsnark'] = pd.Categorical(df1['cpsnark'], categories=custom_order, ordered=True)


        custom_order = ['kzg', 'ipa']
        df1['pc_type'] = pd.Categorical(df1['pc_type'], categories=custom_order, ordered=True)

        # convert to KB
        df1["proof_size_bytes"] = df1["proof_size_bytes"] / 1000.0
        df1["proof_size_bytes"] = df1["proof_size_bytes"].round(0)

        df1.loc[:, "proof_size_bytes"] = df1["proof_size_bytes"].astype(int)

        df_pivot = df1.pivot_table(index=['pc_type', 'model'], columns=['cpsnark'], values='proof_size_bytes') #.reset_index()

        df_pivot = df_pivot.applymap(lambda x: f"{int(x)}" if pd.notnull(x) else "-")


        table_str = df_pivot.to_latex(index=True) \
                        .replace("cpsnark", "") \
                        .replace("model", "") \
                        .replace("pc\_type", "") \
                        .replace("kzg", "\multirow{7}{*}{\rotatebox[origin=c]{90}{KZG}}") \
                        .replace("ipa", "\midrule\n\multirow{7}{*}{\rotatebox[origin=c]{90}{IPA}}") \
                        .replace("lllllll", "llccccc")

        with open(f"{output_dir}/proof_size_table.tex", "w") as f:
            f.write(table_str)