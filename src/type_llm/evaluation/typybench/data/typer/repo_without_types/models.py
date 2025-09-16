import inspect
import io
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import click
import click.shell_completion

if TYPE_CHECKING:  # pragma: no cover
    from .core import TyperCommand, TyperGroup
    from .main import Typer


NoneType = type(None)

AnyType = Type[Any]

Required = ...


class Context(click.Context):
    pass


class FileText(io.TextIOWrapper):
    pass


class FileTextWrite(FileText):
    pass


class FileBinaryRead(io.BufferedReader):
    pass


class FileBinaryWrite(io.BufferedWriter):
    pass


class CallbackParam(click.Parameter):
    pass


class DefaultPlaceholder:
    """
    You shouldn't use this class directly.

    It's used internally to recognize when a default value has been overwritten, even
    if the new value is `None`.
    """

    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return bool(self.value)


DefaultType = TypeVar("DefaultType")

CommandFunctionType = TypeVar("CommandFunctionType", bound=Callable[..., Any])


def Default(value):
    """
    You shouldn't use this function directly.

    It's used internally to recognize when a default value has been overwritten, even
    if the new value is `None`.
    """
    return DefaultPlaceholder(value)  # type: ignore


class CommandInfo:
    def __init__(
        self, name = None, *,
        cls = None, context_settings = None, callback = None, help = None, epilog = None, short_help = None, options_metavar = "[OPTIONS]", add_help_option = True, no_args_is_help = False, hidden = False, deprecated = False, rich_help_panel = None):
        self.name = name
        self.cls = cls
        self.context_settings = context_settings
        self.callback = callback
        self.help = help
        self.epilog = epilog
        self.short_help = short_help
        self.options_metavar = options_metavar
        self.add_help_option = add_help_option
        self.no_args_is_help = no_args_is_help
        self.hidden = hidden
        self.deprecated = deprecated
        # Rich settings
        self.rich_help_panel = rich_help_panel


class TyperInfo:
    def __init__(
        self, typer_instance = Default(None), *,
        name = Default(None), cls = Default(None), invoke_without_command = Default(False), no_args_is_help = Default(False), subcommand_metavar = Default(None), chain = Default(False), result_callback = Default(None), context_settings = Default(None), callback = Default(None), help = Default(None), epilog = Default(None), short_help = Default(None), options_metavar = Default("[OPTIONS]"), add_help_option = Default(True), hidden = Default(False), deprecated = Default(False), rich_help_panel = Default(None)):
        self.typer_instance = typer_instance
        self.name = name
        self.cls = cls
        self.invoke_without_command = invoke_without_command
        self.no_args_is_help = no_args_is_help
        self.subcommand_metavar = subcommand_metavar
        self.chain = chain
        self.result_callback = result_callback
        self.context_settings = context_settings
        self.callback = callback
        self.help = help
        self.epilog = epilog
        self.short_help = short_help
        self.options_metavar = options_metavar
        self.add_help_option = add_help_option
        self.hidden = hidden
        self.deprecated = deprecated
        self.rich_help_panel = rich_help_panel


class ParameterInfo:
    def __init__(
        self, *,
        default = None, param_decls = None, callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, parser = None, click_type = None, show_default = True, show_choices = True, show_envvar = True, help = None, hidden = False, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None):
        # Check if user has provided multiple custom parsers
        if parser and click_type:
            raise ValueError(
                "Multiple custom type parsers provided. "
                "`parser` and `click_type` may not both be provided."
            )

        self.default = default
        self.param_decls = param_decls
        self.callback = callback
        self.metavar = metavar
        self.expose_value = expose_value
        self.is_eager = is_eager
        self.envvar = envvar
        self.shell_complete = shell_complete
        self.autocompletion = autocompletion
        self.default_factory = default_factory
        # Custom type
        self.parser = parser
        self.click_type = click_type
        # TyperArgument
        self.show_default = show_default
        self.show_choices = show_choices
        self.show_envvar = show_envvar
        self.help = help
        self.hidden = hidden
        # Choice
        self.case_sensitive = case_sensitive
        # Numbers
        self.min = min
        self.max = max
        self.clamp = clamp
        # DateTime
        self.formats = formats
        # File
        self.mode = mode
        self.encoding = encoding
        self.errors = errors
        self.lazy = lazy
        self.atomic = atomic
        # Path
        self.exists = exists
        self.file_okay = file_okay
        self.dir_okay = dir_okay
        self.writable = writable
        self.readable = readable
        self.resolve_path = resolve_path
        self.allow_dash = allow_dash
        self.path_type = path_type
        # Rich settings
        self.rich_help_panel = rich_help_panel


class OptionInfo(ParameterInfo):
    def __init__(
        self, *,
        # ParameterInfo
        default = None, param_decls = None, callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, parser = None, click_type = None, show_default = True, prompt = False, confirmation_prompt = False, prompt_required = True, hide_input = False, is_flag = None, flag_value = None, count = False, allow_from_autoenv = True, help = None, hidden = False, show_choices = True, show_envvar = True, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None):
        super().__init__(
            default=default,
            param_decls=param_decls,
            callback=callback,
            metavar=metavar,
            expose_value=expose_value,
            is_eager=is_eager,
            envvar=envvar,
            shell_complete=shell_complete,
            autocompletion=autocompletion,
            default_factory=default_factory,
            # Custom type
            parser=parser,
            click_type=click_type,
            # TyperArgument
            show_default=show_default,
            show_choices=show_choices,
            show_envvar=show_envvar,
            help=help,
            hidden=hidden,
            # Choice
            case_sensitive=case_sensitive,
            # Numbers
            min=min,
            max=max,
            clamp=clamp,
            # DateTime
            formats=formats,
            # File
            mode=mode,
            encoding=encoding,
            errors=errors,
            lazy=lazy,
            atomic=atomic,
            # Path
            exists=exists,
            file_okay=file_okay,
            dir_okay=dir_okay,
            writable=writable,
            readable=readable,
            resolve_path=resolve_path,
            allow_dash=allow_dash,
            path_type=path_type,
            # Rich settings
            rich_help_panel=rich_help_panel,
        )
        self.prompt = prompt
        self.confirmation_prompt = confirmation_prompt
        self.prompt_required = prompt_required
        self.hide_input = hide_input
        self.is_flag = is_flag
        self.flag_value = flag_value
        self.count = count
        self.allow_from_autoenv = allow_from_autoenv


class ArgumentInfo(ParameterInfo):
    def __init__(
        self, *,
        # ParameterInfo
        default = None, param_decls = None, callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, parser = None, click_type = None, show_default = True, show_choices = True, show_envvar = True, help = None, hidden = False, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None):
        super().__init__(
            default=default,
            param_decls=param_decls,
            callback=callback,
            metavar=metavar,
            expose_value=expose_value,
            is_eager=is_eager,
            envvar=envvar,
            shell_complete=shell_complete,
            autocompletion=autocompletion,
            default_factory=default_factory,
            # Custom type
            parser=parser,
            click_type=click_type,
            # TyperArgument
            show_default=show_default,
            show_choices=show_choices,
            show_envvar=show_envvar,
            help=help,
            hidden=hidden,
            # Choice
            case_sensitive=case_sensitive,
            # Numbers
            min=min,
            max=max,
            clamp=clamp,
            # DateTime
            formats=formats,
            # File
            mode=mode,
            encoding=encoding,
            errors=errors,
            lazy=lazy,
            atomic=atomic,
            # Path
            exists=exists,
            file_okay=file_okay,
            dir_okay=dir_okay,
            writable=writable,
            readable=readable,
            resolve_path=resolve_path,
            allow_dash=allow_dash,
            path_type=path_type,
            # Rich settings
            rich_help_panel=rich_help_panel,
        )


class ParamMeta:
    empty = inspect.Parameter.empty

    def __init__(
        self, *,
        name, default = inspect.Parameter.empty, annotation = inspect.Parameter.empty):
        self.name = name
        self.default = default
        self.annotation = annotation


class DeveloperExceptionConfig:
    def __init__(
        self, *,
        pretty_exceptions_enable = True, pretty_exceptions_show_locals = True, pretty_exceptions_short = True):
        self.pretty_exceptions_enable = pretty_exceptions_enable
        self.pretty_exceptions_show_locals = pretty_exceptions_show_locals
        self.pretty_exceptions_short = pretty_exceptions_short
