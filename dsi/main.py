"""The main entrypoint for the CLI."""

import logging
import os

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from dsi.configs.cli import ConfigCLI, RunType
from dsi.configs.experiment.latency import ConfigLatency
from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.configs.table import ConfigTableRecord
from dsi.heatmap.enrich import enrich
from dsi.offline.heatmap.manager import Manager
from dsi.offline.simul.dsi import SimulDSI
from dsi.offline.simul.si import SimulSI
from dsi.online.heatmap.manager import ManagerOnline
from dsi.online.latency.experiment import ExperimentLatency
from dsi.plot.heatmap import PlotHeatmap
from dsi.plot.iters_dist import PlotIters
from dsi.plot.utils import savefig
from dsi.types.heatmap.df_heatmap import DataFrameHeatmap
from dsi.types.heatmap.manager import _Manager
from dsi.types.result import ResultLatency, ResultSimul

log = logging.getLogger(__name__)


def sanity(cfg: ConfigCLI) -> None:
    res_si: ResultSimul = SimulSI(cfg.simul).run()
    log.info("res_si: %s", res_si)
    res_dsi: ResultSimul = SimulDSI(cfg.simul).run()
    log.info("res_dsi: %s", res_dsi)
    log.info("Plotting SI")
    plot_si: PlotIters = PlotIters(
        result=res_si, suptitle=f"Latency of SI (lookahead={cfg.simul.k})"
    )
    plot_si.plot()
    filepath_plots: str = savefig(name="si_latency_and_iters_dist")
    log.info("Figure saved at %s", filepath_plots)


def heatmap(cfg: ConfigCLI) -> None:
    heatmap_filepath: None | str = cfg.load_csv
    if heatmap_filepath is None:
        log.info(
            "Running a new experiment. Results will be stored at %s", heatmap_filepath
        )
        m_cls: _Manager = ManagerOnline if cfg.heatmap.online else Manager
        m = m_cls(config_heatmap=cfg.heatmap, simul_defaults=cfg.simul)
        df_results: pd.DataFrame = m.run()
        df_heatmap: DataFrameHeatmap = enrich(df_results)
        filepath: str = df_heatmap.store()
        log.info("Heatmap stored at %s", filepath)
    else:
        log.info(
            "Received a path to load existing heatmap results."
            " Only visualizing the heatmap rather than running a new experiment."
        )
        log.info("Loading results from %s", heatmap_filepath)
        df_heatmap: DataFrameHeatmap = DataFrameHeatmap.from_heatmap_csv(cfg.load_csv)
    log.info(f"{df_heatmap.shape=}")
    log.info("df_heatmap.head():")
    log.info(df_heatmap.head())
    log.info("df_heatmap.describe():")
    log.info(df_heatmap.describe())
    plot_heatmap = PlotHeatmap(df_heatmap)
    config: ConfigPlotHeatmap
    for config in tqdm(cfg.plots.heatmaps, desc="Plotting heatmaps", unit="plot"):
        log.info(f"Plotting speedup of {config=}")
        filepath: str = plot_heatmap.plot(config)
        log.info("Figure saved at %s", filepath)


def table(cfg: ConfigCLI) -> None:
    record: ConfigTableRecord
    for record in cfg.table.records:
        log.info(
            "Running latency experiments to estimate the latency of computing "
            "a forward pass of off-the-shelf LLMs."
        )
        config_latency: ConfigLatency
        for config_latency in [record.target_latency, record.drafter_latency]:
            log.info(f"Running {config_latency=}")
            res: ResultLatency = ExperimentLatency(config_latency).run()
            log.info(f"Result: {res=}")


@hydra.main(version_base=None, config_name="config")
def main(cfg: ConfigCLI) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Working directory: %s", os.getcwd())
    log.info(
        "Output directory: %s",
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    match cfg.type:
        case RunType.sanity:
            log.info("Running offline simulations of SI and visualizing them.")
            sanity(cfg)
        case RunType.heatmap:
            log.info("Running heatmap. Could be either online or offline.")
            heatmap(cfg)
        case RunType.table:
            log.info(
                "Reproducing the table from the paper."
                " Could be either online or offline."
            )
            table(cfg)
        case _:
            raise NotImplementedError(f"Invalid simulation type: {cfg.type}")
