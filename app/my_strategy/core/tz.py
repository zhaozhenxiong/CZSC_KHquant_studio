"""Timezone helpers pinned to the project's local timezone.

The Windows C runtime ignores IANA timezone names such as ``Asia/Shanghai``
and has no ``time.tzset``, so ``datetime.now()`` silently follows whatever
``TZ`` the launching process inherited (or the OS clock). That once recorded
backtest timestamps seven hours off. These helpers pin persisted timestamps
to an explicit zoneinfo timezone so results are correct regardless of how the
dashboard/CLI was launched.
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DEFAULT_TZ_NAME = "Asia/Shanghai"


def _resolve_zone() -> ZoneInfo:
    name = os.environ.get("KHQUANT_TZ") or _DEFAULT_TZ_NAME
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        # A Windows CRT TZ string such as "CST-8" is not an IANA name; the
        # project is A-share focused, so fall back to China Standard Time.
        return ZoneInfo(_DEFAULT_TZ_NAME)


LOCAL_TZ = _resolve_zone()


def local_now() -> datetime:
    """Aware current datetime in the project's local timezone."""
    return datetime.now(LOCAL_TZ)


def local_now_naive() -> datetime:
    """Naive local wall-clock datetime (offset stripped) for naive storage."""
    return local_now().replace(tzinfo=None)
