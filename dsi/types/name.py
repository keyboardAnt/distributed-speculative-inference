class Name:
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
    speedup_dsi_vs_si = "speedup_fed_vs_spec"
    speedup_dsi_vs_nonsi = "speedup_fed_vs_nonspec"
    speedup_si_vs_nonsi = "speedup_spec_vs_nonspec"


class Param(Name):
    c = "c"
    a = "a"
    k = "k"
    num_target_servers = "num_target_servers"
