"""Option handling and Option management logic."""
from __future__ import annotations
import argparse
import enum
import functools
import logging
from collections.abc import Sequence
from typing import Any
from typing import Callable
from flake8 import utils
from flake8.plugins.finder import Plugins
LOG = logging.getLogger(__name__)
# represent a singleton of "not passed arguments".
# an enum is chosen to trick mypy
_ARG = enum.Enum('_ARG', 'NO')

def _flake8_normalize(value: str, *args: str, comma_separated_list: bool=False, normalize_paths: bool=False) -> str | list[str]:
    ret: str | list[str] = value
    if comma_separated_list and isinstance(ret, str):
        ret = utils.parse_comma_separated_list(value)
    if normalize_paths:
        if isinstance(ret, str):
            ret = utils.normalize_path(ret, *args)
        else:
            ret = utils.normalize_paths(ret, *args)
    return ret

class Option:
    """Our wrapper around an argparse argument parsers to add features."""

    def __init__(self, short_option_name: str | _ARG=_ARG.NO, long_option_name: str | _ARG=_ARG.NO, action: str | type[argparse.Action] | _ARG=_ARG.NO, default: Any | _ARG=_ARG.NO, type: Callable[..., Any] | _ARG=_ARG.NO, dest: str | _ARG=_ARG.NO, nargs: int | str | _ARG=_ARG.NO, const: Any | _ARG=_ARG.NO, choices: Sequence[Any] | _ARG=_ARG.NO, help: str | _ARG=_ARG.NO, metavar: str | _ARG=_ARG.NO, required: bool | _ARG=_ARG.NO, parse_from_config: bool=False, comma_separated_list: bool=False, normalize_paths: bool=False) -> None:
        # Options below are taken from argparse.ArgumentParser.add_argument
        # Options below here are specific to Flake8
        'Initialize an Option instance.\n\n        The following are all passed directly through to argparse.\n\n        :param short_option_name:\n            The short name of the option (e.g., ``-x``). This will be the\n            first argument passed to ``ArgumentParser.add_argument``\n        :param long_option_name:\n            The long name of the option (e.g., ``--xtra-long-option``). This\n            will be the second argument passed to\n            ``ArgumentParser.add_argument``\n        :param default:\n            Default value of the option.\n        :param dest:\n            Attribute name to store parsed option value as.\n        :param nargs:\n            Number of arguments to parse for this option.\n        :param const:\n            Constant value to store on a common destination. Usually used in\n            conjunction with ``action="store_const"``.\n        :param choices:\n            Possible values for the option.\n        :param help:\n            Help text displayed in the usage information.\n        :param metavar:\n            Name to use instead of the long option name for help text.\n        :param required:\n            Whether this option is required or not.\n\n        The following options may be passed directly through to :mod:`argparse`\n        but may need some massaging.\n\n        :param type:\n            A callable to normalize the type (as is the case in\n            :mod:`argparse`).\n        :param action:\n            Any action allowed by :mod:`argparse`.\n\n        The following parameters are for Flake8\'s option handling alone.\n\n        :param parse_from_config:\n            Whether or not this option should be parsed out of config files.\n        :param comma_separated_list:\n            Whether the option is a comma separated list when parsing from a\n            config file.\n        :param normalize_paths:\n            Whether the option is expecting a path or list of paths and should\n            attempt to normalize the paths to absolute paths.\n        '
        if long_option_name is _ARG.NO and short_option_name is not _ARG.NO and short_option_name.startswith('--'):
            short_option_name, long_option_name = (_ARG.NO, short_option_name)
        # flake8 special type normalization
        if comma_separated_list or normalize_paths:
            type = functools.partial(_flake8_normalize, comma_separated_list=comma_separated_list, normalize_paths=normalize_paths)
        self.short_option_name = short_option_name
        self.long_option_name = long_option_name
        self.option_args = [x for x in (short_option_name, long_option_name) if x is not _ARG.NO]
        self.action = action
        self.default = default
        self.type = type
        self.dest = dest
        self.nargs = nargs
        self.const = const
        self.choices = choices
        self.help = help
        self.metavar = metavar
        self.required = required
        self.option_kwargs: dict[str, Any | _ARG] = {'action': self.action, 'default': self.default, 'type': self.type, 'dest': self.dest, 'nargs': self.nargs, 'const': self.const, 'choices': self.choices, 'help': self.help, 'metavar': self.metavar, 'required': self.required}
        # Set our custom attributes
        self.parse_from_config = parse_from_config
        self.comma_separated_list = comma_separated_list
        self.normalize_paths = normalize_paths
        self.config_name: str | None = None
        if parse_from_config:
            if long_option_name is _ARG.NO:
                raise ValueError('When specifying parse_from_config=True, a long_option_name must also be specified.')
            self.config_name = long_option_name[2:].replace('-', '_')
        self._opt = None

    @property
    def filtered_option_kwargs(self) -> dict[str, Any]:
        """Return any actually-specified arguments."""
        return {k: v for k, v in self.option_kwargs.items() if v is not _ARG.NO}

    def __repr__(self) -> str:  # noqa: D105
        parts = []
        for arg in self.option_args:
            parts.append(arg)
        for k, v in self.filtered_option_kwargs.items():
            parts.append(f'{k}={v!r}')
        return f"Option({', '.join(parts)})"

    def normalize(self, value: Any, *normalize_args: str) -> Any:
        """Normalize the value based on the option configuration."""
        if self.comma_separated_list and isinstance(value, str):
            value = utils.parse_comma_separated_list(value)
        if self.normalize_paths:
            if isinstance(value, list):
                value = utils.normalize_paths(value, *normalize_args)
            else:
                value = utils.normalize_path(value, *normalize_args)
        return value

    def to_argparse(self) -> tuple[list[str], dict[str, Any]]:
        """Convert a Flake8 Option to argparse ``add_argument`` arguments."""
        return (self.option_args, self.filtered_option_kwargs)

class OptionManager:
    """Manage Options and OptionParser while adding post-processing."""

    def __init__(self, *, version: str, plugin_versions: str, parents: list[argparse.ArgumentParser], formatter_names: list[str]) -> None:
        """Initialize an instance of an OptionManager."""
        self.formatter_names = formatter_names
        self.parser = argparse.ArgumentParser(prog='flake8', usage='%(prog)s [options] file file ...', parents=parents, epilog=f'Installed plugins: {plugin_versions}')
        self.parser.add_argument('--version', action='version', version=f'{version} ({plugin_versions}) {utils.get_python_version()}')
        self.parser.add_argument('filenames', nargs='*', metavar='filename')
        self.config_options_dict: dict[str, Option] = {}
        self.options: list[Option] = []
        self.extended_default_ignore: list[str] = []
        self.extended_default_select: list[str] = []
        self._current_group: argparse._ArgumentGroup | None = None
    # TODO: maybe make this a free function to reduce api surface area

    def register_plugins(self, plugins: Plugins) -> None:
        """Register the plugin options (if needed)."""
        groups: dict[str, argparse._ArgumentGroup] = {}

        def _set_group(name: str) -> None:
            try:
                self._current_group = groups[name]
            except KeyError:
                group = self.parser.add_argument_group(name)
                self._current_group = groups[name] = group
        for loaded in plugins.all_plugins():
            add_options = getattr(loaded.obj, 'add_options', None)
            if add_options:
                _set_group(loaded.plugin.package)
                add_options(self)
            if loaded.plugin.entry_point.group == 'flake8.extension':
                self.extend_default_select([loaded.entry_name])
        # isn't strictly necessary, but seems cleaner
        self._current_group = None

    def add_option(self, *args: Any, **kwargs: Any) -> None:
        """Create and register a new option.

        See parameters for :class:`~flake8.options.manager.Option` for
        acceptable arguments to this method.

        .. note::

            ``short_option_name`` and ``long_option_name`` may be specified
            positionally as they are with argparse normally.
        """
        option = Option(*args, **kwargs)
        option_args, option_kwargs = option.to_argparse()
        if self._current_group is not None:
            self._current_group.add_argument(*option_args, **option_kwargs)
        else:
            self.parser.add_argument(*option_args, **option_kwargs)
        self.options.append(option)
        if option.parse_from_config:
            name = option.config_name
            assert name is not None
            self.config_options_dict[name] = option
            self.config_options_dict[name.replace('_', '-')] = option
        LOG.debug('Registered option "%s".', option)

    def extend_default_ignore(self, error_codes: Sequence[str]) -> None:
        """Extend the default ignore list with the error codes provided.

        :param error_codes:
            List of strings that are the error/warning codes with which to
            extend the default ignore list.
        """
        LOG.debug('Extending default ignore list with %r', error_codes)
        self.extended_default_ignore.extend(error_codes)

    def extend_default_select(self, error_codes: Sequence[str]) -> None:
        """Extend the default select list with the error codes provided.

        :param error_codes:
            List of strings that are the error/warning codes with which
            to extend the default select list.
        """
        LOG.debug('Extending default select list with %r', error_codes)
        self.extended_default_select.extend(error_codes)

    def parse_args(self, args: Sequence[str] | None=None, values: argparse.Namespace | None=None) -> argparse.Namespace:
        """Proxy to calling the OptionParser's parse_args method."""
        if values:
            self.parser.set_defaults(**vars(values))
        return self.parser.parse_args(args)