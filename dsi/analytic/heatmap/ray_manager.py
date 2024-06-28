import os
from datetime import datetime

import pandas as pd
import ray

from dsi.analytic.heatmap.config_heatmap import Param, get_df_heatmap_params
from dsi.analytic.heatmap.objective import calc_all


class RayManager:
    def __init__(self) -> None:
        self._confs: pd.DataFrame = get_df_heatmap_params()
        self._results_raw: None | list[tuple[int, dict[str, float]]] = None
        self.result: pd.DataFrame = self._confs.copy(deep=True)

    def run(self) -> pd.DataFrame:
        ray.init(
            ignore_reinit_error=True
        )  # Ray discovers and utilizes all available resources by default
        futures = [
            self._process_row.remote(index=index, row=row)
            for index, row in self._confs.iterrows()
        ]
        self._results_raw = ray.get(futures)
        ray.shutdown()
        self._merge_results()
        return self.result

    @staticmethod
    @ray.remote
    def _process_row(index: int, row: dict) -> tuple[int, dict[str, float]]:
        c: float = row[Param.c]
        a: float = row[Param.a]
        k: int = int(row[Param.k])
        all_analytic: dict[str, float] = calc_all(c=c, a=a, k=k)
        return index, all_analytic

    def _merge_results(self) -> None:
        for i, res in self._results_raw:
            for key, val in res.items():
                self.result.at[i, key] = val

    def store(self, dirpath: str) -> None:
        """Store the parsed results in the given directory."""
        if not os.path.exists(dirpath):
            print(f"Creating directory {dirpath}")
            os.makedirs(dirpath)
        now: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename: str = f"heatmap-{now}.csv"
        filepath: str = os.path.join(dirpath, filename)
        self.result.to_csv(filepath)
        print(f"Results stored in {filepath}")
