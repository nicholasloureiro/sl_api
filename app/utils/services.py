from datetime import date, datetime, timedelta
from functools import wraps
import time
from typing import List, Dict, Any
from ..api.schemas.dashboard import FilterRequest, AgeGroup


def query_timer(func):
    """Decorator to measure query execution time"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Add timing to response if it's a dict
        if isinstance(result, dict):
            result.setdefault("query_performance", {})["execution_time_ms"] = round(
                execution_time * 1000, 2
            )

        return result

    return wrapper


def get_clickhouse_date_condition(days_back: int) -> str:
    """Generate ClickHouse compatible date condition"""
    return f"toDate(data_vacina) >= toDate(now()) - INTERVAL {days_back} DAY"


def get_clickhouse_today_condition() -> str:
    """Generate ClickHouse compatible today condition"""
    return "toDate(data_vacina) = toDate(now())"


def apply_date_range_preset(filters: FilterRequest) -> FilterRequest:
    """Apply predefined date range presets"""
    if not filters.date_range_preset:
        return filters

    today = datetime.now().date()

    if filters.date_range_preset == "last_7_days":
        filters.start_date = today - timedelta(days=7)
        filters.end_date = today
    elif filters.date_range_preset == "last_30_days":
        filters.start_date = today - timedelta(days=30)
        filters.end_date = today
    elif filters.date_range_preset == "last_90_days":
        filters.start_date = today - timedelta(days=90)
        filters.end_date = today
    elif filters.date_range_preset == "last_year":
        filters.start_date = today - timedelta(days=365)
        filters.end_date = today
    elif filters.date_range_preset == "ytd":
        filters.start_date = date(today.year, 1, 1)
        filters.end_date = today

    return filters


def build_comprehensive_where_conditions(
    filters: FilterRequest,
) -> tuple[List[str], Dict[str, Any]]:
    """Build comprehensive WHERE conditions with proper parameterization"""
    conditions = []
    params = {}

    # Apply date range presets
    filters = apply_date_range_preset(filters)

    # Date filters
    if filters.start_date:
        conditions.append("toDate(data_vacina) >= %(start_date)s")
        params["start_date"] = filters.start_date.isoformat()

    if filters.end_date:
        conditions.append("toDate(data_vacina) <= %(end_date)s")
        params["end_date"] = filters.end_date.isoformat()

    # Vaccine filters
    if filters.vaccine_types:
        placeholders = ",".join(
            [f"%(vaccine_{i})s" for i in range(len(filters.vaccine_types))]
        )
        conditions.append(f"sigla_vacina IN ({placeholders})")
        for i, vaccine in enumerate(filters.vaccine_types):
            params[f"vaccine_{i}"] = vaccine

    if filters.exclude_vaccine_types:
        placeholders = ",".join(
            [
                f"%(exclude_vaccine_{i})s"
                for i in range(len(filters.exclude_vaccine_types))
            ]
        )
        conditions.append(f"sigla_vacina NOT IN ({placeholders})")
        for i, vaccine in enumerate(filters.exclude_vaccine_types):
            params[f"exclude_vaccine_{i}"] = vaccine

    # Geographic filters
    if filters.states:
        placeholders = ",".join([f"%(state_{i})s" for i in range(len(filters.FilterRequest.states))])
        conditions.append(f"sigla_uf_paciente IN ({placeholders})")
        for i, state in enumerate(filters.states):
            params[f"state_{i}"] = state

    if filters.cities:
        placeholders = ",".join([f"%(city_{i})s" for i in range(len(filters.FilterRequest.cities))])
        conditions.append(f"nome_municipio_paciente IN ({placeholders})")
        for i, city in enumerate(filters.cities):
            params[f"city_{i}"] = city

    if filters.exclude_states:
        placeholders = ",".join(
            [f"%(exclude_state_{i})s" for i in range(len(filters.FilterRequest.exclude_states))]
        )
        conditions.append(f"sigla_uf_paciente NOT IN ({placeholders})")
        for i, state in enumerate(filters.exclude_states):
            params[f"exclude_state_{i}"] = state

    # Demographic filters
    if filters.gender:
        conditions.append("tipo_sexo_paciente = %(gender)s")
        params["gender"] = filters.gender

    if filters.min_age is not None:
        conditions.append("numero_idade_paciente >= %(min_age)s")
        params["min_age"] = filters.min_age

    if filters.max_age is not None:
        conditions.append("numero_idade_paciente <= %(max_age)s")
        params["max_age"] = filters.max_age

    # Age groups
    if filters.age_groups:
        age_conditions = []
        for age_group in filters.age_groups:
            if age_group == AgeGroup.child:
                age_conditions.append("numero_idade_paciente < 18")
            elif age_group == AgeGroup.young_adult:
                age_conditions.append(
                    "(numero_idade_paciente >= 18 AND numero_idade_paciente < 30)"
                )
            elif age_group == AgeGroup.adult:
                age_conditions.append(
                    "(numero_idade_paciente >= 30 AND numero_idade_paciente < 50)"
                )
            elif age_group == AgeGroup.middle_aged:
                age_conditions.append(
                    "(numero_idade_paciente >= 50 AND numero_idade_paciente < 65)"
                )
            elif age_group == AgeGroup.senior:
                age_conditions.append("numero_idade_paciente >= 65")

        if age_conditions:
            conditions.append(f"({' OR '.join(age_conditions)})")

    # Race/color filters
    if filters.race_colors:
        placeholders = ",".join(
            [f"%(race_{i})s" for i in range(len(filters.FilterRequest.race_colors))]
        )
        conditions.append(f"nome_raca_cor_paciente IN ({placeholders})")
        for i, race in enumerate(filters.race_colors):
            params[f"race_{i}"] = race

    # Dose filters
    if filters.dose_numbers:
        placeholders = ",".join(
            [f"%(dose_{i})s" for i in range(len(filters.dose_numbers))]
        )
        conditions.append(f"toInt32OrNull(codigo_dose_vacina) IN ({placeholders})")
        for i, dose in enumerate(filters.dose_numbers):
            params[f"dose_{i}"] = dose

    if filters.only_completed_series:
        conditions.append("toInt32OrNull(codigo_dose_vacina) >= 2")

    if filters.only_boosters:
        conditions.append("toInt32OrNull(codigo_dose_vacina) >= 3")

    # Establishment filters
    if filters.establishment_types:
        placeholders = ",".join(
            [f"%(est_type_{i})s" for i in range(len(filters.FilterRequest.establishment_types))]
        )
        conditions.append(f"codigo_tipo_estabelecimento IN ({placeholders})")
        for i, est_type in enumerate(filters.establishment_types):
            params[f"est_type_{i}"] = est_type

    if filters.establishment_names:
        name_conditions = []
        for i, name in enumerate(filters.establishment_names):
            name_conditions.append(
                f"nome_razao_social_estabelecimento ILIKE %(est_name_{i})s"
            )
            params[f"est_name_{i}"] = f"%{name}%"
        conditions.append(f"({' OR '.join(name_conditions)})")

    # Advanced filters
    if filters.has_maternal_condition is not None:
        if filters.has_maternal_condition:
            conditions.append("codigo_condicao_maternal IS NOT NULL")
        else:
            conditions.append("codigo_condicao_maternal IS NULL")

    if filters.indigenous_only:
        conditions.append("codigo_etnia_indigena_paciente IS NOT NULL")

    return conditions, params
