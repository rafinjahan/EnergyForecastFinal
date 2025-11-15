"""
Weather enrichment helpers.

Loads hourly weather observations (FMI Excel extracts) and joins them with the
Fortum load table for comparable timestamps/locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import pandas as pd

from .fortum_loader import load_fortum_training

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_DATA_DIR = PROJECT_ROOT / "Data"
WEATHER_DATA_ENV_VAR = "ENERGYFORECASTING_WEATHER_DIR"


@dataclass(frozen=True)
class WeatherStationConfig:
    slug: str
    station_name: str
    filename: str
    municipality_keywords: Sequence[str]


ESPOO_TAPIOLA = WeatherStationConfig(
    slug="espoo",
    station_name="Espoo Tapiola",
    filename="Espoo Tapiola_ 1.1.2021 - 30.9.2024_bc039f9b-3b44-40e0-a7fe-9aeb036afb8b.xlsx",
    municipality_keywords=("Espoo",),
)

HELSINKI_KAISANIEMI = WeatherStationConfig(
    slug="helsinki_kaisaniemi",
    station_name="Helsinki Kaisaniemi",
    filename="Helsinki Kaisaniemi_ 1.1.2021 - 30.9.2024_47f8e46c-0d9c-4d3a-8618-94bb4d2c342c.xlsx",
    municipality_keywords=("Uusimaa_Others",),
)

VANTAA_HELSINKI_VANTAA = WeatherStationConfig(
    slug="vantaa_airport",
    station_name="Vantaa Helsinki-Vantaa",
    filename="Vantaa Helsinki-Vantaan lentoasema_ 1.1.2021 - 30.9.2024_9fd70b98-5481-451c-9172-0d2531ddfab8.xlsx",
    municipality_keywords=("Vantaa",),
)

ETELA_POHJANMAA_SEINAJOKI = WeatherStationConfig(
    slug="etela_pohjanmaa_seinajoki",
    station_name="Etelä-Pohjanmaa Seinäjoki Pelmaa",
    filename="Etelä-Pohjanmaa Seinäjoki Pelmaa- 1.1.2021 - 30.9.2024_729c20ed-aba6-49bc-8b57-b9a1d703698d.xlsx",
    municipality_keywords=("Etelä-Pohjanmaa",),
)

ETELA_POHJANMAA_AHTARI = WeatherStationConfig(
    slug="etela_pohjanmaa_ahtari",
    station_name="Etelä-Pohjanmaa Ähtäri",
    filename="Etelä_Pohjanmaa_other Ähtäri.xlsx",
    municipality_keywords=("Etelä-Pohjanmaa",),
)

ETELA_SAVO_MIKKELI = WeatherStationConfig(
    slug="etela_savo_mikkeli",
    station_name="Etelä-Savo Mikkeli",
    filename="Etelä-Savo Mikkeli lentoasema_ 1.1.2021 - 30.9.2024_c9deb697-1706-4204-be28-478feaecd54e.xlsx",
    municipality_keywords=("Etelä-Savo",),
)

JYVASKYLA_AIRPORT = WeatherStationConfig(
    slug="jyvaskyla_airport",
    station_name="Jyväskylä Lentoasema",
    filename="Jyväskylä lentoasema_ 1.1.2021 - 30.9.2024_ede2b8a6-ec86-4ed8-8881-f2ae5097ae15.xlsx",
    municipality_keywords=("Jyväskylä", "Keski-Suomi", "Keski-Suomi_Others"),
)

KANTA_HAME_HAMEENLINNA = WeatherStationConfig(
    slug="kanta_hame_hameenlinna",
    station_name="Kanta-Häme Hämeenlinna Katinen",
    filename="Kanta-Häme Hämeenlinna Katinen_ 1.1.2021 - 30.9.2024_8ed2ac91-0020-402c-884f-b35326d3f431.xlsx",
    municipality_keywords=("Kanta-Häme", "Kanta-Häme_Others"),
)

LAHTI_SOPENKORPI = WeatherStationConfig(
    slug="lahti_sopenkorpi",
    station_name="Lahti Sopenkorpi",
    filename="Lahti Sopenkorpi_ 1.1.2021 - 30.9.2024_b3600d58-f275-4722-ac12-bb4a2fc9411f.xlsx",
    municipality_keywords=("Lahti", "Päijät-Häme", "Päijät-Häme_Others"),
)

LAPPEENRANTA_AIRPORT = WeatherStationConfig(
    slug="lappeenranta_airport",
    station_name="Lappeenranta Lentoasema",
    filename="Lappeenranta lentoasema_ 1.1.2021 - 30.9.2024_84f4a144-0b6f-4ca7-8aea-4edd901d7f64.xlsx",
    municipality_keywords=("Lappeenranta",),
)

LAPPI_INARI = WeatherStationConfig(
    slug="lappi_inari",
    station_name="Lappi Inari Ivalo",
    filename="Lappi Inari Ivalo lentoasema- 1.1.2021 - 30.9.2024_7471801b-137f-45bf-86b9-292905334ccf.xlsx",
    municipality_keywords=("Lappi_Others",),
)

LAPPI_SODANKYLA = WeatherStationConfig(
    slug="lappi_sodankyla",
    station_name="Lappi Sodankylä Tähtelä",
    filename="Lappi Sodankylä Tähtelä- 1.1.2021 - 30.9.2024_b060c208-3e7b-4006-861f-78647f07b66f.xlsx",
    municipality_keywords=("Lappi",),
)

LIPERI_JOENSUU = WeatherStationConfig(
    slug="liperi_joensuu",
    station_name="Liperi Joensuu Lentoasema",
    filename="Liperi Joensuu lentoasema_ 1.1.2021 - 30.9.2024_67d8f935-5e17-4bbe-86b9-f71fc84e1b78.xlsx",
    municipality_keywords=("Joensuu",),
)

POHJOIS_KARJALA_TOHMAJARVI = WeatherStationConfig(
    slug="pohjois_karjala_tohmajarvi",
    station_name="Pohjois-Karjala Tohmajärvi",
    filename="Pohjois-Karjala_others Tohmajärvi Kemie- 1.1.2021 - 30.9.2024_d6a66bb1-88cc-4727-99af-db2c52867870.xlsx",
    municipality_keywords=("Pohjois-Karjala",),
)

POHJOIS_KARJALA_LIEKSA = WeatherStationConfig(
    slug="pohjois_karjala_lieksa",
    station_name="Pohjois-Karjala Lieksa",
    filename="Pohjois-Karjala_others Lieksa Lampela- 1.1.2021 - 30.9.2024_9807f75e-e1c0-46ba-aa91-5511a9ee55d1.xlsx",
    municipality_keywords=("Pohjois-Karjala_Others",),
)

OULU_OULUNSALO = WeatherStationConfig(
    slug="oulu_oulunsalo",
    station_name="Oulu Oulunsalo Pellonpää",
    filename="Oulu Oulunsalo Pellonpää- 1.1.2021 - 30.9.2024_eec95dce-6cb6-4b17-a1c3-383716a90676.xlsx",
    municipality_keywords=("Oulu",),
)

POHJOIS_POHJANMAA_TAIVALKOSKI = WeatherStationConfig(
    slug="pohjois_pohjanmaa_taivalkoski",
    station_name="Pohjois-Pohjanmaa Taivalkoski",
    filename="Pohjois-Pohjanmaa_others Taivalkoski kirkonkylä- 1.1.2021 - 30.9.2024_f5f624f3-de95-4591-b6c3-49a9bc894411.xlsx",
    municipality_keywords=("Pohjois-Pohjanmaa",),
)

POHJOIS_POHJANMAA_PYHAJARVI = WeatherStationConfig(
    slug="pohjois_pohjanmaa_pyhajarvi",
    station_name="Pohjois-Pohjanmaa Pyhäjärvi Ojakylä",
    filename="Pohjois-Pohjanmaa_Others Pyhäjärvi Ojakylä- 1.1.2021 - 30.9.2024_dcde386e-2473-430e-a8d5-f30bb9f2f1b7.xlsx",
    municipality_keywords=("Pohjois-Pohjanmaa_Others",),
)

POHJANMAA_VAASA = WeatherStationConfig(
    slug="pohjanmaa_vaasa",
    station_name="Pohjanmaa Vaasa",
    filename="Pohjanmaa Vaasa lentoasema_ 1.1.2021 - 30.9.2024_011c3848-c16d-466d-b925-c3bac452631b.xlsx",
    municipality_keywords=("Pohjanmaa",),
)

POHJOIS_SAVO_KUOPIO = WeatherStationConfig(
    slug="pohjois_savo_kuopio",
    station_name="Pohjois-Savo Siilinjärvi Kuopio",
    filename="Pohjois-Savo Siilinjärvi Kuopio lentoasema_ 1.1.2021 - 30.9.2024_8d598bc1-14fd-424f-9913-622db6422968.xlsx",
    municipality_keywords=("Pohjois-Savo_Others",),
)

PORI_STATION = WeatherStationConfig(
    slug="pori_rautatieasema",
    station_name="Pori Rautatieasema",
    filename="Pori rautatieasema_ 1.1.2021 - 30.9.2024_bab55314-eced-43bc-9460-e10fce05a118.xlsx",
    municipality_keywords=("Pori", "Varsinais-Suomi_Others"),
)

PIRKANMAA_PIRKKALA = WeatherStationConfig(
    slug="pirkanmaa_pirkkala",
    station_name="Pirkanmaa Pirkkala Tampere",
    filename="Pirkanmaa_Others Pirkkala Tampere lentoasema_ 1.1.2021 - 30.9.2024_0981d871-fcce-4256-b628-a5ae1a36c364.xlsx",
    municipality_keywords=("Pirkanmaa", "Pirkanmaa_Others"),
)

TAMPERE_HARMALA = WeatherStationConfig(
    slug="tampere_harmala",
    station_name="Tampere Härmälä",
    filename="Tampere Härmälä_ 1.1.2021 - 30.9.2024_5976ebcf-ef72-4cd8-ba24-7e9d31476fd4.xlsx",
    municipality_keywords=("Tampere",),
)

ROVANIEMI_STATION = WeatherStationConfig(
    slug="rovaniemi_rautatieasema",
    station_name="Rovaniemi Rautatieasema",
    filename="Rovaniemi rautatieasema- 1.1.2021 - 30.9.2024_c48659cf-1f90-4e7e-be36-72d4cf19fcaa.xlsx",
    municipality_keywords=("Rovaniemi",),
)

ALL_WEATHER_STATIONS: Sequence[WeatherStationConfig] = (
    ESPOO_TAPIOLA,
    HELSINKI_KAISANIEMI,
    VANTAA_HELSINKI_VANTAA,
    ETELA_POHJANMAA_SEINAJOKI,
    ETELA_POHJANMAA_AHTARI,
    ETELA_SAVO_MIKKELI,
    JYVASKYLA_AIRPORT,
    KANTA_HAME_HAMEENLINNA,
    LAHTI_SOPENKORPI,
    LAPPEENRANTA_AIRPORT,
    LAPPI_INARI,
    LAPPI_SODANKYLA,
    LIPERI_JOENSUU,
    POHJOIS_KARJALA_TOHMAJARVI,
    POHJOIS_KARJALA_LIEKSA,
    OULU_OULUNSALO,
    POHJOIS_POHJANMAA_TAIVALKOSKI,
    POHJOIS_POHJANMAA_PYHAJARVI,
    POHJANMAA_VAASA,
    POHJOIS_SAVO_KUOPIO,
    PORI_STATION,
    PIRKANMAA_PIRKKALA,
    TAMPERE_HARMALA,
    ROVANIEMI_STATION,
)


def build_weather_enriched_load(
    station: WeatherStationConfig = ESPOO_TAPIOLA,
) -> pd.DataFrame:
    """
    Convenience wrapper that loads Fortum tables and enriches load data with a
    single weather station.
    """

    frames = load_fortum_training()
    return enrich_consumption_with_weather(
        frames["consumption"], frames["groups"], station, frames.get("prices")
    )


def build_all_weather_enriched_loads(
    stations: Sequence[WeatherStationConfig] | None = None,
) -> pd.DataFrame:
    """Load Fortum tables and enrich every configured weather station.

    Args:
        stations: Optional override for which weather stations to pull. When
                  omitted the helper iterates over ``ALL_WEATHER_STATIONS``.

    Returns:
        A long-form DataFrame containing one row per group/hour/station
        combination.
    """

    frames = load_fortum_training()
    return enrich_consumption_with_all_weather(
        frames["consumption"], frames["groups"], stations
    )


def enrich_consumption_with_weather(
    consumption: pd.DataFrame,
    groups: pd.DataFrame,
    station: WeatherStationConfig = ESPOO_TAPIOLA,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adds weather features from `station` to the long-form load table for
    municipalities that match the station's keywords. Optionally adds price
    features keyed on the Fortum hourly spot price table.
    """

    weather = load_weather_station(station)
    consumption_long = _prepare_consumption_for_station(
        consumption, groups, station
    )

    location_slugs = consumption_long["location_key"].unique().tolist()
    if not location_slugs:
        raise ValueError(
            f"No matching groups found for station {station.station_name}"
        )

    weather_with_locations = pd.concat(
        [weather.assign(location_key=slug) for slug in location_slugs],
        ignore_index=True,
    )

    enriched = consumption_long.merge(
        weather_with_locations,
        on=["measured_at", "location_key"],
        how="inner",
    )
    if prices is not None:
        price_features = _prepare_price_features(prices)
        enriched = enriched.merge(price_features, on="measured_at", how="left")
    return enriched


def enrich_consumption_with_all_weather(
    consumption: pd.DataFrame,
    groups: pd.DataFrame,
    stations: Sequence[WeatherStationConfig] | None = None,
) -> pd.DataFrame:
    """Enrich load table with every configured weather station."""

    selected = ALL_WEATHER_STATIONS if stations is None else stations
    enriched_frames = []
    failures: list[str] = []

    for station in selected:
        try:
            enriched_frames.append(
                enrich_consumption_with_weather(consumption, groups, station)
            )
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"{station.station_name}: {exc}")

    if not enriched_frames:
        joined = "; ".join(failures) if failures else "No stations provided."
        raise ValueError(f"Unable to build weather enrichment: {joined}")

    if failures:
        warnings.warn(
            "Some weather stations were skipped:\n" + "\n".join(failures)
        )

    combined = (
        pd.concat(enriched_frames, ignore_index=True)
        .sort_values(["station_slug", "measured_at", "group_id"])
        .reset_index(drop=True)
    )
    return combined


def load_weather_station(station: WeatherStationConfig) -> pd.DataFrame:
    """
    Load and normalize a weather Excel exported from FMI.
    """

    base_dir = _resolve_weather_data_dir()
    file_path = base_dir / station.filename
    if not file_path.exists():
        raise FileNotFoundError(f"Weather file not found: {file_path}")

    weather = _load_single_weather_file(file_path)

    humidity_path = _find_humidity_companion(file_path)
    if humidity_path is not None:
        humidity = _load_single_weather_file(humidity_path)[
            ["measured_at", "humidity_pct"]
        ]
        weather = weather.merge(
            humidity, on="measured_at", how="left", suffixes=("", "_humidity")
        )
        humidity_fallback = weather.pop("humidity_pct_humidity")
        weather["humidity_pct"] = weather["humidity_pct"].combine_first(
            humidity_fallback
        )

    weather = (
        weather.assign(
            station_slug=station.slug, station_name=station.station_name
        )
        .sort_values("measured_at")
        .reset_index(drop=True)
    )
    return weather


def _build_weather_timestamp(df: pd.DataFrame) -> pd.Series:
    local_dt = pd.to_datetime(
        df["Vuosi"].astype(int).astype(str)
        + "-"
        + df["Kuukausi"].astype(int).astype(str).str.zfill(2)
        + "-"
        + df["Päivä"].astype(int).astype(str).str.zfill(2)
        + " "
        + df["Aika [Paikallinen aika]"].astype(str).str.zfill(5),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    return local_dt.dt.tz_localize(
        "Europe/Helsinki", ambiguous="infer", nonexistent="shift_forward"
    )


def _prepare_consumption_for_station(
    consumption: pd.DataFrame,
    groups: pd.DataFrame,
    station: WeatherStationConfig,
) -> pd.DataFrame:
    groups = groups.copy()
    groups["location_key"] = groups["group_label"].apply(_extract_location_key)

    target_slugs = _slugify_many(station.municipality_keywords)
    if not target_slugs:
        raise ValueError(
            f"Station {station.station_name} has no municipality keywords"
        )

    matching_groups = groups[groups["location_key"].isin(target_slugs)]
    if matching_groups.empty:
        raise ValueError(
            f"No groups found for keywords {station.municipality_keywords}"
        )

    group_ids = matching_groups["group_id"].tolist()
    selection_cols = ["measured_at"] + group_ids
    load_subset = consumption[selection_cols].copy()
    load_subset["measured_at"] = pd.to_datetime(
        load_subset["measured_at"], utc=True
    )

    melted = load_subset.melt(
        id_vars="measured_at", var_name="group_id", value_name="load_mwh"
    )
    melted["group_id"] = melted["group_id"].astype(int)

    merged = melted.merge(
        matching_groups[["group_id", "location_key"]], on="group_id", how="left"
    )
    if merged["location_key"].isna().any():
        fallback_slug = target_slugs[0] if target_slugs else station.slug
        merged["location_key"] = merged["location_key"].fillna(fallback_slug)
    return _add_time_features(merged)


def _extract_location_key(group_label: str) -> str:
    parts = [part.strip() for part in group_label.split("|")]
    municipality = parts[2] if len(parts) >= 3 else parts[-1]
    return _slugify(municipality)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _slugify_many(keywords: Iterable[str]) -> Sequence[str]:
    return [_slugify(keyword) for keyword in keywords]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_single_weather_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    timestamp = _build_weather_timestamp(df)
    normalized = pd.DataFrame(
        {
            "measured_at": timestamp.dt.tz_convert("UTC"),
            "temperature_c": _extract_numeric_column(
                df, "Lämpötilan keskiarvo [°C]"
            ),
            "wind_speed_ms": _extract_numeric_column(
                df, "Keskituulen nopeus [m/s]"
            ),
            "precip_mm": _extract_numeric_column(
                df, "Tunnin sademäärä [mm]"
            ),
            "humidity_pct": _extract_numeric_column(
                df, "Suhteellisen kosteuden keskiarvo [%]"
            ),
        }
    )

    return (
        normalized.dropna(subset=["measured_at"])
        .sort_values("measured_at")
        .reset_index(drop=True)
    )


def _extract_numeric_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(pd.NA, index=df.index)
    return _coerce_numeric(df[column_name])


def _find_humidity_companion(file_path: Path) -> Path | None:
    base_normalized = _normalize_filename_for_match(file_path.stem)
    if not base_normalized:
        return None

    for candidate in file_path.parent.glob("*.xlsx"):
        if candidate == file_path:
            continue
        if "humidity" not in candidate.name.lower():
            continue
        candidate_normalized = _normalize_filename_for_match(candidate.stem)
        if candidate_normalized == base_normalized:
            return candidate
    return None


def _normalize_filename_for_match(name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    ascii_name = ascii_name.lower().replace("humidity", "")
    return re.sub(r"[^a-z0-9]+", "", ascii_name)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append deterministic calendar features derived from the measured_at timestamp.
    """

    augmented = df.copy()
    measured = augmented["measured_at"]
    if measured.dt.tz is None:
        measured = measured.dt.tz_localize("UTC")

    local = measured.dt.tz_convert("Europe/Helsinki")
    iso = local.dt.isocalendar()

    augmented["hour"] = local.dt.hour.astype(int)
    augmented["weekday"] = local.dt.weekday.astype(int)
    augmented["is_weekend"] = local.dt.weekday.isin([5, 6]).astype(int)
    augmented["month"] = local.dt.month.astype(int)
    augmented["weekofyear"] = iso.week.astype(int)
    augmented["year"] = local.dt.year.astype(int)
    return augmented


def _prepare_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Adds current spot price plus a 24-hour lag, filling missing lags by
    hour-of-day medians so Oct 2 rows retain informative values.
    """

    df = prices.copy()
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    df = df.sort_values("measured_at").rename(
        columns={"eur_per_mwh": "price_eur_per_mwh"}
    )
    df["price_lag_24h"] = df["price_eur_per_mwh"].shift(24)

    missing = df["price_lag_24h"].isna()
    if missing.any():
        hours = df["measured_at"].dt.tz_convert("Europe/Helsinki").dt.hour
        hourly_median = (
            df.assign(hour=hours)
            .groupby("hour")["price_eur_per_mwh"]
            .transform("median")
        )
        df.loc[missing, "price_lag_24h"] = hourly_median[missing]

    keep = ["measured_at", "price_eur_per_mwh", "price_lag_24h"]
    return df[keep]


def _resolve_weather_data_dir() -> Path:
    override = os.getenv(WEATHER_DATA_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return WEATHER_DATA_DIR
