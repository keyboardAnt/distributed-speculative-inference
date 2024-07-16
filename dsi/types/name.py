from enum import Enum

from pydantic import BaseModel


class Name(str, Enum):
    @classmethod
    def get_all_valid_values(cls):
        return [
            getattr(cls, attr)
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        ]


class HeatmapColumn(Name):
    cost_nonsi = "cost_nonsi"
    cost_si = "cost_si"
    cost_dsi = "cost_dsi"
    cost_baseline = "cost_baseline"
    min_cost_si = "min_cost_si"
    min_cost_dsi = "min_cost_dsi"
    min_cost_baseline = "min_cost_baseline"
    speedup_dsi_vs_si = "speedup_dsi_vs_si"
    speedup_dsi_vs_nonsi = "speedup_dsi_vs_nonsi"
    speedup_si_vs_nonsi = "speedup_si_vs_nonsi"
    min_speedup_dsi_vs_si = "min_speedup_dsi_vs_si"
    min_speedup_dsi_vs_nonsi = "min_speedup_dsi_vs_nonsi"
    min_speedup_si_vs_nonsi = "min_speedup_si_vs_nonsi"
    min_speedup_dsi_vs_baseline = "min_speedup_dsi_vs_baseline"


class Param(Name):
    c = "c"
    a = "a"
    k = "k"
    num_target_servers = "num_target_servers"


class Print(BaseModel):
    name_to_print: dict[Param | HeatmapColumn, str] = {
        Param.c: "Drafter Latency",
        Param.a: "Acceptance Rate",
        Param.k: "Lookahead",
        HeatmapColumn.speedup_dsi_vs_si: "DSI Speedup over SI (x)",
        HeatmapColumn.speedup_dsi_vs_nonsi: "DSI Speedup over non-SI (x)",
        HeatmapColumn.min_speedup_dsi_vs_si: "DSI Speedup over SI (x)",
        HeatmapColumn.min_speedup_dsi_vs_nonsi: "DSI Speedup over non-SI (x)",
        HeatmapColumn.min_speedup_si_vs_nonsi: "SI Speedup over non-SI (x)",
    }
