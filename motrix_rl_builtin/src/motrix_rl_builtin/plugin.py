from motrix_rl import frameworks
from motrix_rl_builtin.fastsac.framework import MotrixFramework


def register() -> None:
    frameworks.register_framework(MotrixFramework())
