"""Shows the library's UserError as a Gradio error message instead of a traceback."""

import functools
import inspect

import gradio as gr

from utils.errors import UserError


def shows_errors(fn):
    """Wrap an event handler (function or generator) so a UserError becomes a gr.Error."""
    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def gen(*args, **kwargs):
            try:
                yield from fn(*args, **kwargs)
            except UserError as e:
                raise gr.Error(str(e)) from e

        return gen

    @functools.wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except UserError as e:
            raise gr.Error(str(e)) from e

    return call
