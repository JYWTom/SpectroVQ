"""Errors about what the user asked for, shown to them as they are.

The library raises UserError for a missing file, a bad setting or a refused request; front ends
turn it into their own message (the GUI into a gr.Error, see gui/errors.py) instead of a
traceback. Anything else is a bug and keeps its traceback.
"""


class UserError(Exception):
    """A problem with a request: its message is written for the person who made it."""
