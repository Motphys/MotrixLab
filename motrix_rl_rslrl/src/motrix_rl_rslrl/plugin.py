from motrix_rl import frameworks
from motrix_rl_rslrl.framework import RslrlFramework


def register() -> None:
    frameworks.register_framework(RslrlFramework())
