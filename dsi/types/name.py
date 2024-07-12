from enum import Enum


class Name(str, Enum):
    @classmethod
    def get_all_valid_values(cls):
        return [
            getattr(cls, attr)
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        ]


class HeatmapColumn(Name):
    cost_nonsi = "cost_nonspec"
    cost_si = "cost_spec"
    cost_dsi = "cost_fed"
    cost_baseline = "cost_baseline"
    min_cost_si = "min_cost_spec"
    min_cost_dsi = "min_cost_fed"
    min_cost_baseline = "min_cost_baseline"
    speedup_dsi_vs_si = "speedup_fed_vs_spec"
    speedup_dsi_vs_nonsi = "speedup_fed_vs_nonspec"
    speedup_si_vs_nonsi = "speedup_spec_vs_nonspec"
    min_speedup_dsi_vs_si = "min_speedup_fed_vs_spec"
    min_speedup_dsi_vs_nonsi = "min_speedup_fed_vs_nonspec"
    min_speedup_si_vs_nonsi = "min_speedup_spec_vs_nonspec"
    min_speedup_dsi_vs_baseline = "min_speedup_fed_vs_baseline"


class Param(Name):
    c = "c"
    a = "a"
    k = "k"
    num_target_servers = "num_target_servers"
