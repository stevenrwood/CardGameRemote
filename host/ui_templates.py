"""
Frontend HTML templates.

Each page the HTTP handler serves lives as its own file in
``host/templates/`` and is loaded from disk at import time so edits
show up on the next request without a server restart (just bump a
page version if you want to force a cache bust in the browser).

Templates that need server-side values use ``string.Template`` with
``$name`` placeholders — plain ``$`` used in regexes or jQuery is
left alone by ``safe_substitute``. CSS / JS braces pass through
unchanged because we no longer rely on f-strings.
"""

from pathlib import Path
from string import Template


TEMPLATES_DIR = Path(__file__).parent / "templates"


def _read(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text()


# Plain strings (no server-side interpolation)
TABLE_HTML = _read("table.html")
LOGVIEW_HTML = _read("logview.html")
CONSOLE_HTML = _read("console.html")

# Templates (need .safe_substitute at render time)
SCANNER_TMPL = Template(_read("scanner.html"))
CALIBRATE_TMPL = Template(_read("calibrate.html"))
