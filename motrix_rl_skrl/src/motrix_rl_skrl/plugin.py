from motrix_rl import frameworks
from motrix_rl_skrl.framework import SkrlFramework


def register() -> None:
    frameworks.register_framework(SkrlFramework())
