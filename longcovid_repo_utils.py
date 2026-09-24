"""Shared utilities for the Long COVID analysis repository.

This helper keeps the analysis scripts self-contained by bundling the small set
of data-cleaning and diary-aggregation functions that were previously imported
from the antioxidant study helper modules.
"""

from __future__ import annotations

import pandas as pd


def replace_empty_with_none(df):
    """Replace empty strings in the DataFrame with None."""
    if df is None:
        return df
    for col_name in df.columns:
        df[col_name] = df[col_name].replace(['', 'NA', 'N/A', 'na', 'NaN'], None)
    return df


def load_raw_data(csv_path):
    """Load raw data from a CSV file."""
    return pd.read_csv(csv_path)


def extract_randomization_data(df):
    """Extract and clean randomization data."""
    random_df = df[df['redcap_event_name'] == 'Randomization'].copy()
    random_df = random_df.dropna(axis=1, how='all')
    return replace_empty_with_none(random_df)


# ---------------------------------------------------------------------------
# Recovery and symptom-alleviation helpers used across analysis scripts
# ---------------------------------------------------------------------------

def sustain_change_to_1(series, set_date=15):
    """Return the first day when a binary outcome becomes sustained 1."""
    series_reset = series.reset_index(drop=True)

    if series_reset.empty:
        return None

    ones_positions = series_reset[series_reset == 1].index

    if len(ones_positions) == 0:
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    for pos in ones_positions:
        subsequent_values = series_reset[pos + 1:]
        if len(subsequent_values) == 0:
            return pos + 1

        valid_subsequent = subsequent_values.isna() | (subsequent_values == 1)
        if valid_subsequent.all():
            return pos + 1

    return set_date if len(ones_positions) > 0 else None


def first_change_to_1(series, set_date=15):
    """Return the first day when a binary outcome changes to 1."""
    series_reset = series.reset_index(drop=True)

    if series_reset.empty:
        return None

    if len(series_reset[series_reset == 1].index) == 0:
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    return series_reset[series_reset == 1].index.min() + 1


def ret_indexof1(series):
    """Return the 1-based index of the first occurrence of 1 in the series."""
    series_reset = series.reset_index(drop=True)
    pos = series_reset[series_reset == 1].index.tolist()
    pos_zero = series_reset[series_reset == 0].index.tolist()
    return pos, pos_zero


def ret_series(series):
    """Return the series as-is."""
    return series.to_list()


def ret_first_alleviation(series, set_date=15):
    """Return the first day when the symptom score drops to <= 1."""
    series_reset = series.reset_index(drop=True)
    if series_reset.empty:
        return None

    alleviation_positions = series_reset[series_reset <= 1].index
    if len(alleviation_positions) == 0:
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    return alleviation_positions.min() + 1


def ret_sustain_alleviation(series, set_date=15):
    """Return the first day when a symptom score remains at or below 1."""
    series_reset = series.reset_index(drop=True)
    if series_reset.empty:
        return None

    alleviation_positions = series_reset[series_reset <= 1].index
    if len(alleviation_positions) == 0:
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    for pos in alleviation_positions:
        subsequent_values = series_reset[pos + 1:]
        if len(subsequent_values) == 0:
            return pos + 1
        valid_subsequent = subsequent_values.isna() | (subsequent_values <= 1)
        if valid_subsequent.all():
            return pos + 1

    return set_date if len(alleviation_positions) > 0 else None
