from typing import IO, Any, Mapping, Optional, Sequence, Union

from click.testing import CliRunner as ClickCliRunner  # noqa
from click.testing import Result
from typer.main import Typer
from typer.main import get_command as _get_command


class CliRunner(ClickCliRunner):
    def invoke(  # type: ignore
        self, app, args = None, input = None, env = None, catch_exceptions = True, color = False, **extra: Any,
    ):
        use_cli = _get_command(app)
        return super().invoke(
            use_cli,
            args=args,
            input=input,
            env=env,
            catch_exceptions=catch_exceptions,
            color=color,
            **extra,
        )
