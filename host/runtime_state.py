"""Single-slot state holder.

The host has exactly one ``AppState`` instance per process — built
in ``main.main()`` and read by every other module. Pulling that
slot into its own module breaks the legacy "every module imports
from overhead_test for `_state`" pattern and keeps the import
graph one-directional.

Use ``runtime_state._state`` for reads/writes (the value is
re-bound by main); avoid ``from runtime_state import _state``
because that takes a snapshot of the binding and won't see the
later reassignment.
"""

_state = None
