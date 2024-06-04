"""The main entrypoint for the CLI."""
import fire

from dsi.main import main


if __name__ == "__main__":
    fire.Fire(main)
