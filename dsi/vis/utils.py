from pathlib import Path

import hydra
from matplotlib import pyplot as plt


def savefig(fig: plt.Figure, filename: str) -> None:
    """
    Stores the figure in the output directory with the given filename.
    The stored figures can be used in the paper.

    Args:
        fig: The figure to store.
        filename: The filename of the stored figure, without the extension.
    """
    dirpath = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    filepath: Path = dirpath / filename
    fig.savefig(
        filepath.with_suffix(".pdf"), dpi=300, format="pdf", bbox_inches="tight"
    )
