from enum import StrEnum, auto

import math
import re

from gluonts.time_feature import get_lags_for_frequency, get_seasonality
from pandas.tseries.frequencies import to_offset


class Setting(StrEnum):
    UNIVARIATE = auto()


class Prior(StrEnum):
    SN = auto()
    ISO = auto()
    OU = auto()
    SE = auto()
    PE = auto()


def _to_canonical_offset(freq: str):
    freq_str = str(freq)
    hourly_match = re.fullmatch(r"(\d*)H", freq_str)
    if hourly_match is not None:
        freq_str = f"{hourly_match.group(1)}h"
    return to_offset(freq_str)


def _offset_signature(freq: str):
    offset = _to_canonical_offset(freq)
    try:
        return ("fixed", int(offset.nanos))
    except ValueError:
        name = getattr(offset, "name", None) or getattr(offset, "rule_code", None) or type(offset).__name__
        return (str(name).lower(), int(getattr(offset, "n", 1)))


def frequencies_match(lhs: str, rhs: str) -> bool:
    return _offset_signature(lhs) == _offset_signature(rhs)


def get_season_length(freq: str) -> int:
    if frequencies_match(freq, "H"):
        return 24
    if frequencies_match(freq, "B"):
        return 30
    if frequencies_match(freq, "1D"):
        return 30
    relative_step_hours = get_relative_time_step(freq)
    if 0.0 < relative_step_hours < 24.0:
        daily_steps = 24.0 / relative_step_hours
        if math.isclose(daily_steps, round(daily_steps), rel_tol=1e-9, abs_tol=1e-9):
            return int(round(daily_steps))
    return int(get_seasonality(str(freq)))


def get_relative_time_step(freq: str) -> float:
    offset = _to_canonical_offset(freq)
    try:
        return float(offset.nanos) / (60.0 * 60.0 * 1e9)
    except ValueError:
        signature = str(_offset_signature(freq)[0]).lower()
        if signature.startswith("b"):
            return 24.0
        if signature.startswith("w"):
            return 24.0 * 7.0
        if signature.startswith("m"):
            return 24.0 * 30.0
        if signature.startswith("q"):
            return 24.0 * 30.0 * 3.0
        if signature.startswith("y") or signature.startswith("a"):
            return 24.0 * 365.0
        raise NotImplementedError(f"Relative time step for {freq} is not implemented yet.")


_CANONICAL_TIME_UNITS_HOURS = [
    1.0,
    24.0,
    24.0 * 7.0,
    24.0 * 30.0,
    24.0 * 30.0 * 3.0,
    24.0 * 365.0,
]


def _next_canonical_unit(window_hours: float) -> float | None:
    for unit_hours in _CANONICAL_TIME_UNITS_HOURS:
        if unit_hours > window_hours:
            return unit_hours
    return None


def _canonical_unit_after(unit_hours: float) -> float | None:
    for candidate in _CANONICAL_TIME_UNITS_HOURS:
        if candidate > unit_hours:
            return candidate
    return None


def get_lags_for_freq(
    freq_str: str,
    *,
    context_length: int | None = None,
    max_multiple: int | None = None,
    max_lag: int | None = None,
):
    def _clip_lags(lags: list[int]) -> list[int]:
        lag_set = {int(lag) for lag in lags if int(lag) > 0}
        if max_lag is not None:
            lag_set = {lag for lag in lag_set if lag <= int(max_lag)}
        elif max_multiple is not None and context_length is not None:
            lag_set = {lag for lag in lag_set if lag <= int(max_multiple) * int(context_length)}
        return sorted(lag_set)

    if context_length is not None:
        context_length = int(context_length)
        if context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {context_length}.")

        if frequencies_match(freq_str, "H"):
            return _clip_lags([context_length * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]])

        if frequencies_match(freq_str, "1D"):
            return _clip_lags([context_length * i for i in [1, 2, 3, 4, 5, 6, 7]])

        if frequencies_match(freq_str, "B"):
            return _clip_lags([context_length * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

        dt_hours = get_relative_time_step(freq_str)
        window_hours = dt_hours * context_length

        first_unit_hours = _next_canonical_unit(window_hours)
        if first_unit_hours is None:
            first_unit_hours = window_hours

        dense_count = max(1, int(first_unit_hours // window_hours))
        lag_steps = {context_length * i for i in range(1, dense_count + 1)}

        second_horizon_hours = _canonical_unit_after(first_unit_hours)
        if second_horizon_hours is None:
            second_horizon_hours = 4.0 * first_unit_hours

        total_anchor_units = max(1, int(second_horizon_hours // first_unit_hours))
        anchor_step_units = max(1, int(math.ceil(total_anchor_units / 4.0)))

        sparse_anchor_units = list(range(anchor_step_units, total_anchor_units, anchor_step_units))
        sparse_anchor_units.append(total_anchor_units)

        for anchor_units in sparse_anchor_units:
            lag_hours = anchor_units * first_unit_hours
            lag_steps.add(max(1, int(round(lag_hours / dt_hours))))

        return _clip_lags(list(lag_steps))

    if frequencies_match(freq_str, "H"):
        return [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]

    if frequencies_match(freq_str, "B"):
        return [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

    if frequencies_match(freq_str, "1D"):
        return [30 * i for i in [1, 2, 3, 4, 5, 6, 7]]

    lags_seq = [int(lag) for lag in get_lags_for_frequency(str(freq_str)) if int(lag) > 0]
    if not lags_seq:
        raise NotImplementedError(f"Lags for {freq_str} are not implemented yet.")
    return sorted(set(lags_seq))
