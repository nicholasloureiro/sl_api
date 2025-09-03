from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from datetime import date


class SortOrder(str, Enum):
    asc = "ASC"
    desc = "DESC"


class TimeGranularity(str, Enum):
    day = "day"
    week = "week"
    month = "month"
    quarter = "quarter"
    year = "year"


class ChartType(str, Enum):
    line = "line"
    bar = "bar"
    pie = "pie"
    area = "area"
    scatter = "scatter"
    heatmap = "heatmap"


class AgeGroup(str, Enum):
    child = "0-17"
    young_adult = "18-29"
    adult = "30-49"
    middle_aged = "50-64"
    senior = "65+"


# Enhanced Models
class FilterRequest(BaseModel):
    """Comprehensive filter model for all endpoints"""

    # Date filters
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    date_range_preset: Optional[str] = Field(
        None, pattern="^(last_7_days|last_30_days|last_90_days|last_year|ytd)$"
    )

    # Vaccine filters
    vaccine_types: Optional[List[str]] = None
    exclude_vaccine_types: Optional[List[str]] = None

    # Geographic filters
    states: Optional[List[str]] = None
    cities: Optional[List[str]] = None
    exclude_states: Optional[List[str]] = None

    # Demographic filters
    gender: Optional[str] = Field(None, pattern="^[MF]$")
    min_age: Optional[int] = Field(None, ge=0, le=135)
    max_age: Optional[int] = Field(None, ge=0, le=135)
    age_groups: Optional[List[AgeGroup]] = None
    race_colors: Optional[List[str]] = None

    # Dose filters
    dose_numbers: Optional[List[int]] = None
    only_completed_series: Optional[bool] = False
    only_boosters: Optional[bool] = False

    # Establishment filters
    establishment_types: Optional[List[int]] = None
    establishment_names: Optional[List[str]] = None
    exclude_establishments: Optional[List[str]] = None

    # Advanced filters
    has_maternal_condition: Optional[bool] = None
    indigenous_only: Optional[bool] = None

    @field_validator("end_date")
    @classmethod
    def end_date_must_be_after_start_date(cls, v, info):
        if info.data.get("start_date") and v:
            if v < info.data["start_date"]:
                raise ValueError("end_date must be after start_date")
        return v


class PaginationRequest(BaseModel):
    cursor: Optional[str] = None
    limit: int = Field(100, ge=1, le=2000)
    sort_by: str = "data_vacina"
    sort_order: SortOrder = SortOrder.desc


class PaginationResponse(BaseModel):
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    metadata: Dict[str, Any]


class ChartData(BaseModel):
    chart_type: ChartType
    title: str
    data: List[Dict[str, Any]]
    labels: Optional[List[str]] = None
    datasets: Optional[List[Dict[str, Any]]] = None
    options: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class DashboardResponse(BaseModel):
    summary_stats: Dict[str, Any]
    charts: List[ChartData]
    filters_applied: Dict[str, Any]
    data_freshness: Dict[str, Any]
    query_performance: Dict[str, Any]


class FilterStats(BaseModel):
    available_vaccines: List[Dict[str, Any]]
    available_states: List[Dict[str, Any]]
    available_cities: List[Dict[str, Any]]
    available_establishments: List[Dict[str, Any]]
    date_range: Dict[str, str]
    age_range: Dict[str, int]
    total_records: int
    data_quality: Dict[str, Any]
