"""
Finnish holiday data loader using holiday-calendar.fi API.

This module fetches Finnish public holidays and adds holiday-related features
to improve energy consumption forecasting.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, Set

import pandas as pd
import requests


class FinnishHolidayLoader:
    """Fetch and cache Finnish holidays from holiday-calendar.fi API."""
    
    API_BASE = "https://holiday-calendar.fi/api/non-working-days"
    RATE_LIMIT_DELAY = 0.5  # API allows 2 req/sec, be conservative
    
    def __init__(self):
        self._cache: Dict[int, Set[str]] = {}
        
    def get_holidays(self, year: int) -> Set[str]:
        """
        Get Finnish public holidays for a specific year (excluding weekends).
        
        Args:
            year: Year to fetch holidays for
            
        Returns:
            Set of date strings in YYYY-MM-DD format
        """
        if year in self._cache:
            return self._cache[year]
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        try:
            url = f"{self.API_BASE}?start={start_date}&end={end_date}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Filter to only actual holidays (not weekends)
            holidays = {
                date_str for date_str, info in data.items()
                if info.get("desc") != "weekend"
            }
            
            self._cache[year] = holidays
            time.sleep(self.RATE_LIMIT_DELAY)  # Respect rate limit
            
            return holidays
            
        except Exception as e:
            print(f"⚠ Failed to fetch holidays for {year}: {e}")
            return set()
    
    def get_holidays_range(self, start_date: str, end_date: str) -> Set[str]:
        """
        Get holidays for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Set of holiday date strings
        """
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        all_holidays = set()
        for year in range(start_year, end_year + 1):
            all_holidays.update(self.get_holidays(year))
        
        # Filter to date range
        return {
            h for h in all_holidays
            if start_date <= h <= end_date
        }


def add_holiday_features(df: pd.DataFrame, date_column: str = "measured_at") -> pd.DataFrame:
    """
    Add Finnish holiday-related features to a DataFrame.
    
    Features added:
    - is_holiday: Boolean for public holidays
    - is_holiday_eve: Day before a holiday
    - is_holiday_after: Day after a holiday
    - days_to_next_holiday: Days until next holiday
    - days_since_holiday: Days since last holiday
    - holiday_week: Week containing a holiday
    
    Args:
        df: DataFrame with datetime column
        date_column: Name of the datetime column
        
    Returns:
        DataFrame with added holiday features
    """
    df = df.copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Get date range
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    
    # Add buffer for eve/after features
    start_date = (min_date - timedelta(days=2)).strftime("%Y-%m-%d")
    end_date = (max_date + timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Fetch holidays
    print(f"  Fetching Finnish holidays from {start_date[:4]} to {end_date[:4]}...")
    loader = FinnishHolidayLoader()
    holidays = loader.get_holidays_range(start_date, end_date)
    print(f"  ✓ Fetched {len(holidays)} holidays")
    
    # Convert to datetime set for comparison
    holiday_dates = {pd.to_datetime(h).date() for h in holidays}
    
    # Extract date from datetime
    df_dates = df[date_column].dt.date
    
    # Basic holiday indicator
    df["is_holiday"] = df_dates.isin(holiday_dates).astype(int)
    
    # Holiday eve (day before)
    df["is_holiday_eve"] = df_dates.apply(
        lambda d: (d + timedelta(days=1)) in holiday_dates
    ).astype(int)
    
    # Day after holiday
    df["is_holiday_after"] = df_dates.apply(
        lambda d: (d - timedelta(days=1)) in holiday_dates
    ).astype(int)
    
    # Days to next holiday (capped at 30)
    def days_to_next_holiday(date):
        future_holidays = [h for h in holiday_dates if h > date]
        if future_holidays:
            next_holiday = min(future_holidays)
            return min((next_holiday - date).days, 30)
        return 30
    
    df["days_to_next_holiday"] = df_dates.apply(days_to_next_holiday)
    
    # Days since last holiday (capped at 30)
    def days_since_last_holiday(date):
        past_holidays = [h for h in holiday_dates if h < date]
        if past_holidays:
            last_holiday = max(past_holidays)
            return min((date - last_holiday).days, 30)
        return 30
    
    df["days_since_last_holiday"] = df_dates.apply(days_since_last_holiday)
    
    # Week containing a holiday
    df["holiday_week"] = 0
    for holiday in holiday_dates:
        # Mark entire week (Mon-Sun) as holiday week
        week_start = holiday - timedelta(days=holiday.weekday())
        week_end = week_start + timedelta(days=6)
        mask = (df_dates >= week_start) & (df_dates <= week_end)
        df.loc[mask, "holiday_week"] = 1
    
    # Major holiday periods (Christmas, Midsummer, Easter)
    df["is_major_holiday_period"] = 0
    
    # Christmas period (Dec 20 - Jan 6)
    christmas_mask = (
        ((df[date_column].dt.month == 12) & (df[date_column].dt.day >= 20)) |
        ((df[date_column].dt.month == 1) & (df[date_column].dt.day <= 6))
    )
    df.loc[christmas_mask, "is_major_holiday_period"] = 1
    
    # Midsummer period (around June 20-26)
    midsummer_mask = (
        (df[date_column].dt.month == 6) & 
        (df[date_column].dt.day >= 19) & 
        (df[date_column].dt.day <= 27)
    )
    df.loc[midsummer_mask, "is_major_holiday_period"] = 1
    
    return df


def get_finnish_holidays_simple(years: list[int]) -> Set[str]:
    """
    Simple function to get Finnish holidays for multiple years.
    
    Args:
        years: List of years to fetch
        
    Returns:
        Set of holiday date strings in YYYY-MM-DD format
    """
    loader = FinnishHolidayLoader()
    all_holidays = set()
    
    for year in years:
        all_holidays.update(loader.get_holidays(year))
    
    return all_holidays


if __name__ == "__main__":
    # Test the module
    print("Testing Finnish Holiday Loader...")
    
    holidays_2024 = get_finnish_holidays_simple([2024])
    print(f"\nFinnish holidays in 2024: {len(holidays_2024)}")
    print("Examples:")
    for h in sorted(holidays_2024)[:10]:
        print(f"  - {h}")
    
    # Test with a DataFrame
    print("\nTesting with sample DataFrame...")
    df = pd.DataFrame({
        "measured_at": pd.date_range("2024-01-01", "2024-01-31", freq="D")
    })
    
    df_with_holidays = add_holiday_features(df)
    print(f"\nFeatures added: {[c for c in df_with_holidays.columns if c != 'measured_at']}")
    print("\nSample with holidays:")
    print(df_with_holidays[df_with_holidays["is_holiday"] == 1])
