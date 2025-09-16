from typing import Literal, TypedDict


class SystemPrompts(TypedDict):
    html_css = None
    html_tailwind = None
    react_tailwind = None
    bootstrap = None
    ionic_tailwind = None
    vue_tailwind = None
    svg = None


Stack = Literal[
    "html_css",
    "html_tailwind",
    "react_tailwind",
    "bootstrap",
    "ionic_tailwind",
    "vue_tailwind",
    "svg",
]
