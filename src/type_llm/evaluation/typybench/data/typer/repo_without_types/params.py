from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type, Union, overload

import click

from .models import ArgumentInfo, OptionInfo

if TYPE_CHECKING:  # pragma: no cover
    import click.shell_completion


# Overload for Option created with custom type 'parser'
@overload
def Option(
    # Parameter
    default = ..., *param_decls: str,
    callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, parser = None, show_default = True, prompt = False, confirmation_prompt = False, prompt_required = True, hide_input = False, is_flag = None, flag_value = None, count = False, allow_from_autoenv = True, help = None, hidden = False, show_choices = True, show_envvar = True, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None): ...


# Overload for Option created with custom type 'click_type'
@overload
def Option(
    # Parameter
    default = ..., *param_decls: str,
    callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, click_type = None, show_default = True, prompt = False, confirmation_prompt = False, prompt_required = True, hide_input = False, is_flag = None, flag_value = None, count = False, allow_from_autoenv = True, help = None, hidden = False, show_choices = True, show_envvar = True, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None): ...


def Option(
    # Parameter
    default = ..., *param_decls: str,
    callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, parser = None, click_type = None, show_default = True, prompt = False, confirmation_prompt = False, prompt_required = True, hide_input = False, is_flag = None, flag_value = None, count = False, allow_from_autoenv = True, help = None, hidden = False, show_choices = True, show_envvar = True, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None):
    return OptionInfo(
        # Parameter
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
        # Option
        show_default=show_default,
        prompt=prompt,
        confirmation_prompt=confirmation_prompt,
        prompt_required=prompt_required,
        hide_input=hide_input,
        is_flag=is_flag,
        flag_value=flag_value,
        count=count,
        allow_from_autoenv=allow_from_autoenv,
        help=help,
        hidden=hidden,
        show_choices=show_choices,
        show_envvar=show_envvar,
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


# Overload for Argument created with custom type 'parser'
@overload
def Argument(
    # Parameter
    default = ..., *,
    callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, parser = None, show_default = True, show_choices = True, show_envvar = True, help = None, hidden = False, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None): ...


# Overload for Argument created with custom type 'click_type'
@overload
def Argument(
    # Parameter
    default = ..., *,
    callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, click_type = None, show_default = True, show_choices = True, show_envvar = True, help = None, hidden = False, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None): ...


def Argument(
    # Parameter
    default = ..., *,
    callback = None, metavar = None, expose_value = True, is_eager = False, envvar = None, shell_complete = None, autocompletion = None, default_factory = None, parser = None, click_type = None, show_default = True, show_choices = True, show_envvar = True, help = None, hidden = False, case_sensitive = True, min = None, max = None, clamp = False, formats = None, mode = None, encoding = None, errors = "strict", lazy = None, atomic = False, exists = False, file_okay = True, dir_okay = True, writable = False, readable = True, resolve_path = False, allow_dash = False, path_type = None, rich_help_panel = None):
    return ArgumentInfo(
        # Parameter
        default=default,
        # Arguments can only have one param declaration
        # it will be generated from the param name
        param_decls=None,
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
