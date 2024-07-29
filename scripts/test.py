import argparse
import subprocess
from enum import Enum


class TestMarker(Enum):
    SERIAL = "serial"
    NOTSERIAL = "not serial"
    ALL = "all"  # Adding 'all' as a valid TestMarker


def run_pytest(marker: TestMarker, pytest_args):
    """Helper method to run pytest with given markers and additional arguments."""
    if marker in [TestMarker.SERIAL, TestMarker.NOTSERIAL]:
        n_option = "0" if marker == TestMarker.SERIAL else "auto"
        cmd = ["pytest", "-m", marker.value, "-n", n_option] + pytest_args
        print("Executing command:", " ".join(cmd))
        subprocess.run(cmd)
    elif marker == TestMarker.ALL:
        print("Running all tests...")
        run_pytest(TestMarker.SERIAL, pytest_args)
        run_pytest(TestMarker.NOTSERIAL, pytest_args)


def main():
    parser = argparse.ArgumentParser(description="Run pytest with different modes.")
    parser.add_argument(
        "mode",
        choices=[marker.name.lower() for marker in TestMarker],
        help="The mode to run tests in.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest",
    )

    args = parser.parse_args()

    marker = TestMarker[args.mode.upper()]
    run_pytest(marker, args.pytest_args)


if __name__ == "__main__":
    main()
