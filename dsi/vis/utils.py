from pathlib import Path

import hydra
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def savefig(name: str, fig: None | Figure = None) -> str:
    """
    Stores the figure in the output directory with the given filename.
    Returns the path to the stored figure.
    The stored figures can be used in the paper.

    Args:
        filename: The filename of the stored figure, without the extension.
        fig: The figure to store. If None, the current figure is stored.
    """
    if fig is None:
        fig = plt.gcf()
    plt.tight_layout()
    dirpath = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    filepath: Path = (dirpath / name).with_suffix(".pdf")
    fig.savefig(filepath, dpi=300, format="pdf", bbox_inches="tight")
    return str(filepath)
