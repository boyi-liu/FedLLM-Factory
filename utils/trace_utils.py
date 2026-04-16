"""
Utilities for parsing device state traces and extracting trainable time windows.

A trainable window requires all three conditions to hold simultaneously:
  - Screen is off AND locked
  - Device is charging (battery_charged_on)
  - Connected to WiFi
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_TRACE_FILE = Path(__file__).parent / "state_traces.json"

_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_events(text: str) -> List[tuple]:
    """Parse tab-separated log lines into (timestamp, event) pairs."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        ts_str, event = parts[0].strip(), parts[1].strip()
        try:
            ts = datetime.strptime(ts_str, _DATE_FMT)
        except ValueError:
            continue
        events.append((ts, event))
    return events


def get_trainable_windows(client_id: Union[str, int]) -> List[dict]:
    """
    Return the trainable time windows for a given client.

    A window is active when **all** of the following hold simultaneously:
      - screen is off AND locked
      - device is charging
      - device is on WiFi

    Parameters
    ----------
    client_id : str or int
        The client identifier used as the top-level key in state_traces.json.

    Returns
    -------
    list of dict
        Each element has keys ``"start"`` and ``"end"``, both ``datetime`` objects.

    Raises
    ------
    KeyError
        If *client_id* is not found in the trace file.
    """
    if not _TRACE_FILE.exists():
        raise FileNotFoundError(f"Trace file not found: {_TRACE_FILE}")

    with open(_TRACE_FILE, encoding="utf-8") as f:
        data = json.load(f)

    key = str(client_id)
    if key not in data:
        raise KeyError(f"Client '{client_id}' not found in {_TRACE_FILE}")

    raw_text = "".join(data[key]["messages"])
    events = _parse_events(raw_text)

    # --- state tracking ---
    screen_off = False
    screen_locked = False
    charging = False
    on_wifi = False

    window_start: Optional[datetime] = None
    windows: List[dict] = []

    def _all_conditions_met() -> bool:
        return (screen_off or screen_locked) and charging and on_wifi

    for ts, event in events:
        was_trainable = _all_conditions_met()

        if event == "screen_off":
            screen_off = True
        elif event == "screen_on":
            screen_off = False
        elif event == "screen_lock":
            screen_locked = True
        elif event == "screen_unlock":
            screen_locked = False
        elif event == "battery_charged_on":
            charging = True
        elif event == "battery_charged_off":
            charging = False
        elif event == "wifi":
            on_wifi = True
        elif event in ("2G", "Unknown"):
            on_wifi = False
        else:
            # battery percentage snapshots and other events — skip
            continue

        is_trainable = _all_conditions_met()

        if not was_trainable and is_trainable:
            # window opens
            window_start = ts
        elif was_trainable and not is_trainable:
            # window closes
            if window_start is not None:
                if (ts - window_start).total_seconds() >= 60:
                    windows.append({"start": window_start, "end": ts})
                window_start = None

    # close any still-open window at the last event timestamp
    if _all_conditions_met() and window_start is not None and events:
        last_ts = events[-1][0]
        if (last_ts - window_start).total_seconds() >= 60:
            windows.append({"start": window_start, "end": last_ts})

    return windows

def plot_trainable_windows(
    n: int = 5,
    save_path: Optional[str] = None,
    time_range: Optional[timedelta] = None,
) -> None:
    """
    Visualize trainable windows for the first *n* clients as a Gantt-style timeline.

    Parameters
    ----------
    n : int
        Number of clients to include (default 5).
    save_path : str, optional
        If given, save the figure to this path instead of displaying it.
    time_range : timedelta, optional
        If given, only show windows whose end time falls within this duration
        from the global earliest start time across all clients.
        E.g. ``timedelta(hours=24)`` keeps only the first 24 hours of activity.
    """
    with open(_TRACE_FILE, encoding="utf-8") as f:
        data = json.load(f)

    client_ids = list(data.keys())[:n]

    # Collect all windows up-front so we can compute the global earliest time.
    all_windows = {cid: get_trainable_windows(cid) for cid in client_ids}

    global_earliest: Optional[datetime] = None
    if time_range is not None:
        for windows in all_windows.values():
            for w in windows:
                if global_earliest is None or w["start"] < global_earliest:
                    global_earliest = w["start"]

    fig, ax = plt.subplots(figsize=(14, max(3, n * 0.6)))

    for row, cid in enumerate(client_ids):
        windows = all_windows[cid]
        for w in windows:
            if (
                time_range is not None
                and global_earliest is not None
                and w["end"] - global_earliest > time_range
            ):
                continue
            start = mdates.date2num(w["start"])
            end = mdates.date2num(w["end"])
            ax.barh(row, end - start, left=start, height=0.6, align="center",
                    color="steelblue", alpha=0.8)

    ax.set_yticks(range(len(client_ids)))
    ax.set_yticklabels([f"Client {cid}" for cid in client_ids])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.set_xlabel("Time")
    ax.set_title(f"Trainable Windows — First {len(client_ids)} Clients")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # windows = get_trainable_windows(0)
    # for w in windows:
    #     print(f"  Start: {w['start']}, End: {w['end']}")
    plot_trainable_windows(n=10, time_range=timedelta(hours=24), save_path='./trace.png')
        