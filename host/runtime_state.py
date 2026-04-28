"""Single-slot state holder.

The host has exactly one ``AppState`` instance per process —
built in ``main.main()`` and read by every other module. Lives
in its own one-line module so the import graph stays
one-directional: everything imports from runtime_state, and
runtime_state imports nothing.

Use ``runtime_state._state`` for reads/writes (the value is
re-bound by main); avoid ``from runtime_state import _state``
because that takes a snapshot of the binding and won't see the
later reassignment.
"""

_state = None
