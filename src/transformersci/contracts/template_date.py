"""Freeze the date chat templates see, so a contract's prompt is the same every day.

A chat template can call `strftime_now(...)`, which Transformers implements as
`datetime.now().strftime(...)` in `transformers.utils.chat_template_utils`. Such a
template (Granite 3.1, Muse, Mistral3, Llama 3.2 Vision, ...) puts today's date
into the prompt, so a baseline only matches the day it was recorded. The runner
freezes that module's clock to `TEMPLATE_DATE` before any contract runs, in its own
process and in every isolated contract process, and records the date in each
baseline it writes.

Nothing else sees the frozen clock: only the module's own `datetime` name is
replaced, and only once the module is imported, so loading order is unchanged.
"""

import datetime
import importlib.abc
import sys

TEMPLATE_DATE = "2026-01-01"
MODULE = "transformers.utils.chat_template_utils"


def frozen_datetime(value):
    moment = datetime.datetime.fromisoformat(value)

    class FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.combine(moment.date(), moment.time(), tzinfo=tz)

    return FrozenDatetime


def patch(module, value):
    if isinstance(getattr(module, "datetime", None), type):
        module.datetime = frozen_datetime(value)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Patches the module right after it is first imported, without importing it early."""

    def __init__(self, value):
        self.value = value

    def find_spec(self, name, path=None, target=None):
        if name != MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader, value = spec.loader, self.value
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module, value)

        loader.exec_module = exec_and_patch
        return spec


def freeze(value=TEMPLATE_DATE):
    """Make `strftime_now` in chat templates report `value` (ISO date) from now on."""
    if not value:
        return
    datetime.date.fromisoformat(value[:10])  # fail loudly on a malformed date
    if MODULE in sys.modules:
        patch(sys.modules[MODULE], value)
    hook = next((f for f in sys.meta_path if isinstance(f, _PatchOnImport)), None)
    if hook:
        hook.value = value
    else:
        sys.meta_path.insert(0, _PatchOnImport(value))
