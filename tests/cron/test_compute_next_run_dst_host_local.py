"""Regression: ``compute_next_run`` must compute each cron slot with the zone's DST RULES, not
the UTC offset that is current when the slot is computed.

The configured-zone path (``HERMES_TIMEZONE`` / config ``timezone`` / a per-job pin) already
did this. The server-local path (no zone configured) did not: the base instant then carries a
fixed ``datetime.timezone`` offset (``datetime.now().astimezone()``), the next slot was stamped
with that same offset, and the first run after a clock change fired an hour early (autumn) or
late (spring). Seen 2026-10-25 for a daily heartbeat in a DST zone.

Behaviour across transitions (both modes, PEP 495 / zoneinfo semantics):

* skipped spring hour: a slot whose wall clock does not exist (01:30 on a 01:00->02:00 day)
  fires once, shifted forward by the gap (02:30 summer time), never skipped and never doubled;
* repeated autumn hour: a fixed-time slot inside it fires once, at its FIRST occurrence
  (summer time); the next run is the next day, not the second 01:30.
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron.jobs import compute_next_run

DAILY_9 = {"kind": "cron", "expr": "0 9 * * *"}
DAILY_0130 = {"kind": "cron", "expr": "30 1 * * *"}


def _set_host_tz(name):
    if name is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = name
    time.tzset()


@pytest.fixture(params=["host-local", "configured"])
def zone_mode(request, monkeypatch):
    """Run each case with the zone as the HOST's local zone (nothing configured) and as the
    configured profile zone; returns a setter taking the IANA name."""
    saved_tz = os.environ.get("TZ")

    def configure(name):
        if request.param == "host-local":
            monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
            _set_host_tz(name)
        else:
            monkeypatch.setenv("HERMES_TIMEZONE", name)
            _set_host_tz("UTC")
        hermes_time.reset_cache()
        if request.param == "host-local":
            assert hermes_time.get_timezone() is None
        return ZoneInfo(name)

    yield configure
    _set_host_tz(saved_tz)
    hermes_time.reset_cache()


def _next(schedule, last_run_at):
    return datetime.fromisoformat(compute_next_run(schedule, last_run_at=last_run_at))


@pytest.mark.parametrize("zone, last, expected", [
    # Europe/London: spring forward 2026-03-29 01:00 GMT, fall back 2026-10-25 02:00 BST.
    ("Europe/London", "2026-03-28T09:00:00+00:00", "2026-03-29T09:00:00+01:00"),
    ("Europe/London", "2026-10-24T09:00:00+01:00", "2026-10-25T09:00:00+00:00"),
    # America/New_York: spring forward 2026-03-08, fall back 2026-11-01.
    ("America/New_York", "2026-03-07T09:00:00-05:00", "2026-03-08T09:00:00-04:00"),
    ("America/New_York", "2026-10-31T09:00:00-04:00", "2026-11-01T09:00:00-05:00"),
])
def test_daily_slot_after_a_transition_keeps_the_wall_clock(zone_mode, zone, last, expected):
    zone_mode(zone)
    nxt = _next(DAILY_9, last)
    assert nxt == datetime.fromisoformat(expected)
    assert nxt.utcoffset() == datetime.fromisoformat(expected).utcoffset()


def test_full_year_walk_never_drifts(zone_mode):
    tz = zone_mode("America/New_York")
    last = datetime(2026, 1, 1, 9, 0, tzinfo=tz)
    for _ in range(365):
        nxt = _next(DAILY_9, last.isoformat())
        wall = nxt.astimezone(tz)
        assert (wall.hour, wall.minute) == (9, 0), f"drift after {last.isoformat()}: {nxt}"
        assert (nxt - last).total_seconds() in (23 * 3600, 24 * 3600, 25 * 3600)
        last = nxt


def test_slot_in_the_skipped_spring_hour_fires_once_shifted_forward(zone_mode):
    zone_mode("Europe/London")
    # 2026-03-29 01:30 does not exist in London; zoneinfo (fold=0) maps it to 02:30 BST.
    first = _next(DAILY_0130, "2026-03-28T01:30:00+00:00")
    assert first == datetime.fromisoformat("2026-03-29T02:30:00+01:00")
    second = _next(DAILY_0130, first.isoformat())
    assert second == datetime.fromisoformat("2026-03-30T01:30:00+01:00")


def test_slot_in_the_repeated_autumn_hour_fires_once_at_first_occurrence(zone_mode):
    zone_mode("Europe/London")
    first = _next(DAILY_0130, "2026-10-24T01:30:00+01:00")
    assert first == datetime.fromisoformat("2026-10-25T01:30:00+01:00")  # BST occurrence
    second = _next(DAILY_0130, first.isoformat())
    assert second == datetime.fromisoformat("2026-10-26T01:30:00+00:00")  # not 01:30 GMT
