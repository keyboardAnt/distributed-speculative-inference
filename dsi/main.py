"""The main entrypoint for the CLI."""


class SimulationMode(str):
    analytic = "analytic"
    thread_pool = "thread_pool"


def main(simulation_mode: SimulationMode):
    print(f"{simulation_mode=}")
    print("Starting...")
