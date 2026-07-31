import argparse
from typing import Any, Dict, List, Optional, Tuple


class Parser:
    """Build a small argparse wrapper from keyword defaults."""

    def __init__(self, description: str = "Training configuration parser", **kwargs: Any):
        self.parser = argparse.ArgumentParser(description=description)
        self._add_arguments(**kwargs)

    def _add_arguments(self, **kwargs: Any) -> None:
        for name, default in kwargs.items():
            option_name = f"--{name.replace('_', '-')}"
            self.parser.add_argument(option_name, default=default)

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        return self.parser.parse_args(args)


    def get_defaults(self) -> Dict[str, Any]:
        return {key: value for key, value in self.parser.parse_args([]).__dict__.items()}
