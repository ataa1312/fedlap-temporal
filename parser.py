import sys
from argparse import Namespace, ArgumentParser

import yaml
from configs.registry import Registry
from configs.config import get_default_config, overlay_config


class Parser:
    def __init__(self) -> None:
        self._parser = self._build()

    # ----------------------------- public API ----------------------------- #

    def parse_args(self, argv: list[str] | None = None) -> Namespace:
        if argv is None and len(sys.argv) == 1:
            self._parser.print_help()
            sys.exit(1)
        return self._parser.parse_args(argv)

    def load_config(self, args: Namespace) -> Registry:
        # Start from the default tree and overlay the (possibly partial) YAML,
        # then apply --set overrides. Unlike Registry.from_yaml (full replace),
        # this lets YAMLs specify only what differs from the defaults.
        config = get_default_config()
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        overlay_config(config, data)
        self.apply_overrides(config, args.overrides)
        return config

    # ----------------------------- helpers -------------------------------- #

    @staticmethod
    def parse_value(raw: str):
        """Coerce a CLI string into a Python value via YAML rules.

        PyYAML follows YAML 1.1, which requires a decimal point before the
        exponent ('1.0e-3' parses as float, '1e-3' does not). We post-process
        strings that look like floats so '--set lr=1e-3' works as users expect.
        """
        val = yaml.safe_load(raw)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                pass
        return val

    @staticmethod
    def apply_overrides(config: Registry, pairs: list[str]) -> None:
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(
                    f"Override must be of the form key=value, got: {pair!r}"
                )
            key, raw = pair.split("=", 1)
            config.set_path(key, Parser.parse_value(raw))

    # ----------------------------- internals ------------------------------ #

    def _build(self) -> ArgumentParser:
        parser = ArgumentParser(
            add_help=True,
            exit_on_error=True,
            description="Train a federated model from a YAML config with optional overrides.",
        )
        parser.add_argument(
            "-c", "--config", help="Path to the YAML config file.", required=True
        )
        parser.add_argument(
            "-r",
            "--repeat",
            type=int,
            default=1,
            metavar="N",
            help="Repeat the run with N consecutive seeds (default: 1).",
        )
        parser.add_argument(
            "-l",
            "--logs",
            default=None,
            metavar="PATH",
            help="Optional path to a log file.",
        )
        parser.add_argument(
            "-L",
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging verbosity (default: INFO).",
        )
        parser.add_argument(
            "--resume",
            default=None,
            metavar="DIR",
            help=(
                "Resume a run from an existing output dir (reads its ckpt/). "
                "Use the exact seed-level dir, e.g. results/<cfg>-<stamp>/seed-1234. "
                "Skips minting a fresh timestamped dir and forces auto_resume."
            ),
        )
        parser.add_argument(
            "--set",
            dest="overrides",
            nargs="+",
            action="extend",
            default=[],
            metavar="KEY=VALUE",
            help=(
                "Override config entries by dot path. Can be passed multiple "
                "times; values accumulate. Examples:\n"
                "  --set train.num_epochs=50 dataset.directed=true\n"
                "  --set train.num_epochs=50 --set dataset.directed=true"
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()
    config = parser.load_config(args)
    print(config)
