from fastapi import APIRouter
from fastapi import HTTPException, Query
from ..database.session import ClickHouseDep
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time
from .schemas.dashboard import (
    FilterRequest, DashboardResponse, ChartData, ChartType,
    FilterStats, PaginationRequest, PaginationResponse, TimeGranularity,
)
from ..utils.services import (
    query_timer, build_comprehensive_where_conditions,
    get_clickhouse_date_condition, get_clickhouse_today_condition,
)
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
def root():
    return {"ok": True}


@router.post("/dashboard/overview")
@query_timer
async def get_dashboard_overview(
    client: ClickHouseDep,
    filters: FilterRequest = FilterRequest(),
    include_charts: bool = True,
) -> DashboardResponse:
    """Get comprehensive dashboard overview with all key metrics and charts"""
    try:
        conditions, params = build_comprehensive_where_conditions(filters)
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Summary statistics query
        summary_query = f"""
        SELECT 
            count(*) as total_vaccinations,
            uniq(codigo_paciente) as unique_patients,
            uniq(sigla_vacina) as unique_vaccines,
            uniq(sigla_uf_paciente) as unique_states,
            uniq(nome_municipio_paciente) as unique_cities,
            uniq(codigo_cnes_estabelecimento) as unique_establishments,
            
            -- Gender distribution
            countIf(tipo_sexo_paciente = 'F') as female_count,
            countIf(tipo_sexo_paciente = 'M') as male_count,
            
            -- Age statistics
            round(avg(numero_idade_paciente), 2) as avg_age,
            min(numero_idade_paciente) as min_age,
            max(numero_idade_paciente) as max_age,
            
            -- Dose distribution
            countIf(toInt32OrNull(codigo_dose_vacina) = 1) as first_doses,
            countIf(toInt32OrNull(codigo_dose_vacina) = 2) as second_doses,
            countIf(toInt32OrNull(codigo_dose_vacina) >= 3) as boosters,
            
            -- Date range
            min(data_vacina) as earliest_date,
            max(data_vacina) as latest_date,
            
            -- Age groups
            countIf(numero_idade_paciente < 18) as children,
            countIf(numero_idade_paciente >= 18 AND numero_idade_paciente < 30) as young_adults,
            countIf(numero_idade_paciente >= 30 AND numero_idade_paciente < 50) as adults,
            countIf(numero_idade_paciente >= 50 AND numero_idade_paciente < 65) as middle_aged,
            countIf(numero_idade_paciente >= 65) as seniors
        FROM events 
        {where_clause}
        """

        result = client.query(summary_query, params)
        summary_data = (
            dict(zip(result.column_names, result.result_rows[0]))
            if result.result_rows
            else {}
        )

        # Calculate derived metrics
        total_vaccinations = summary_data.get("total_vaccinations", 0)
        unique_patients = summary_data.get("unique_patients", 0)

        summary_stats = {
            **summary_data,
            "completion_rate": round(
                (summary_data.get("second_doses", 0) / unique_patients * 100), 2
            )
            if unique_patients > 0
            else 0,
            "booster_rate": round(
                (summary_data.get("boosters", 0) / unique_patients * 100), 2
            )
            if unique_patients > 0
            else 0,
            "doses_per_person": round(total_vaccinations / unique_patients, 2)
            if unique_patients > 0
            else 0,
            "geographic_coverage_rate": round(
                (summary_data.get("unique_cities", 0) / 5570 * 100), 2
            ),
        }

        charts = []

        if include_charts:
            # Chart 1: Vaccination trends over time (last 30 days)
            trend_conditions = conditions.copy()
            trend_conditions.append(get_clickhouse_date_condition(30))
            trend_where_clause = (
                "WHERE " + " AND ".join(trend_conditions) if trend_conditions else ""
            )

            trends_query = f"""
            SELECT 
                toDate(data_vacina) as date,
                count(*) as daily_count,
                uniq(codigo_paciente) as unique_patients
            FROM events 
            {trend_where_clause}
            GROUP BY toDate(data_vacina)
            ORDER BY date
            """

            result = client.query(trends_query, params)
            trend_data = [
                dict(zip(result.column_names, row)) for row in result.result_rows
            ]

            charts.append(
                ChartData(
                    chart_type=ChartType.line,
                    title="Daily Vaccination Trends (Last 30 Days)",
                    data=trend_data,
                    options={"responsive": True, "interaction": {"intersect": False}},
                    metadata={"total_days": len(trend_data)},
                )
            )

            # Chart 2: Vaccine distribution
            vaccine_query = f"""
            SELECT 
                sigla_vacina as vaccine,
                count(*) as count,
                round(count(*) * 100.0 / sum(count(*)) OVER(), 2) as percentage
            FROM events 
            {where_clause}
            GROUP BY sigla_vacina
            ORDER BY count DESC
            LIMIT 10
            """

            result = client.query(vaccine_query, params)
            vaccine_data = [
                dict(zip(result.column_names, row)) for row in result.result_rows
            ]

            charts.append(
                ChartData(
                    chart_type=ChartType.pie,
                    title="Vaccine Distribution",
                    data=vaccine_data,
                    options={
                        "responsive": True,
                        "plugins": {"legend": {"position": "right"}},
                    },
                    metadata={"total_vaccines": len(vaccine_data)},
                )
            )

            # Chart 3: Age group distribution
            age_data = [
                {"age_group": "0-17", "count": summary_stats.get("children", 0)},
                {"age_group": "18-29", "count": summary_stats.get("young_adults", 0)},
                {"age_group": "30-49", "count": summary_stats.get("adults", 0)},
                {"age_group": "50-64", "count": summary_stats.get("middle_aged", 0)},
                {"age_group": "65+", "count": summary_stats.get("seniors", 0)},
            ]

            charts.append(
                ChartData(
                    chart_type=ChartType.bar,
                    title="Age Group Distribution",
                    data=age_data,
                    options={
                        "responsive": True,
                        "scales": {"y": {"beginAtZero": True}},
                    },
                    metadata={"total_age_groups": 5},
                )
            )

        return DashboardResponse(
            summary_stats=summary_stats,
            charts=charts,
            filters_applied=filters.model_dump(exclude_unset=True),
            data_freshness={
                "last_updated": datetime.now().isoformat(),
                "data_range": {
                    "start": str(summary_stats.get("earliest_date", "")),
                    "end": str(summary_stats.get("latest_date", "")),
                },
            },
            query_performance={"total_queries": 4 + len(charts), "cached": False},
        )

    except Exception as e:
        logger.error(f"Dashboard overview error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")


# =============================================================================
# ENHANCED FILTERING AND METADATA
# =============================================================================


@router.get("/filters/metadata", response_model=FilterStats)
@query_timer
async def get_comprehensive_filter_metadata(client: ClickHouseDep) -> FilterStats:
    """Get comprehensive filter metadata with counts and data quality metrics"""
    try:
        # Get vaccines
        vaccines_query = """
        SELECT 
            sigla_vacina as value,
            count(*) as count,
            uniq(codigo_paciente) as unique_patients
        FROM events
        WHERE sigla_vacina != '' AND sigla_vacina IS NOT NULL
        GROUP BY sigla_vacina
        ORDER BY count DESC
        """

        result = client.query(vaccines_query)
        available_vaccines = [
            {
                "value": row[0],
                "label": row[0],
                "count": row[1],
                "unique_patients": row[2],
            }
            for row in result.result_rows
        ]

        # Get states
        states_query = """
        SELECT 
            sigla_uf_estabelecimento as value,
            nome_uf_estabelecimento as name,
            count(*) as count,
            uniq(nome_municipio_estabelecimento) as cities_count
        FROM events
        WHERE sigla_uf_estabelecimento != '' AND sigla_uf_estabelecimento IS NOT NULL
        GROUP BY sigla_uf_estabelecimento, nome_uf_estabelecimento
        ORDER BY count DESC
        """

        result = client.query(states_query)
        available_states = [
            {
                "value": row[0],
                "label": f"{row[1]} ({row[0]})",
                "count": row[2],
                "cities_count": row[3],
            }
            for row in result.result_rows
        ]

        # Get cities
        cities_query = """
        SELECT 
            nome_municipio_estabelecimento as value,
            sigla_uf_estabelecimento as state,
            count(*) as count
        FROM events
        WHERE nome_municipio_estabelecimento != '' AND nome_municipio_estabelecimento IS NOT NULL
        GROUP BY nome_municipio_estabelecimento, sigla_uf_estabelecimento
        ORDER BY count DESC
        LIMIT 200
        """

        result = client.query(cities_query)
        available_cities = [
            {
                "value": row[0],
                "label": f"{row[0]}, {row[1]}",
                "state": row[1],
                "count": row[2],
            }
            for row in result.result_rows
        ]

        # Get establishments
        establishments_query = """
        SELECT 
            codigo_tipo_estabelecimento as code,
            descricao_tipo_estabelecimento as description,
            count(*) as count,
            uniq(codigo_cnes_estabelecimento) as establishments_count
        FROM events
        WHERE codigo_tipo_estabelecimento IS NOT NULL
        GROUP BY codigo_tipo_estabelecimento, descricao_tipo_estabelecimento
        ORDER BY count DESC
        """

        result = client.query(establishments_query)
        available_establishments = [
            {
                "code": row[0],
                "description": row[1],
                "count": row[2],
                "establishments_count": row[3],
            }
            for row in result.result_rows
        ]

        # General stats and data quality
        general_query = """
        SELECT 
            count(*) as total_records,
            min(data_vacina) as min_date,
            max(data_vacina) as max_date,
            min(numero_idade_paciente) as min_age,
            max(numero_idade_paciente) as max_age,
            
            -- Data quality metrics
            countIf(codigo_paciente IS NULL OR codigo_paciente = '') as missing_patient_id,
            countIf(data_vacina IS NULL) as missing_date,
            countIf(sigla_vacina IS NULL OR sigla_vacina = '') as missing_vaccine,
            countIf(numero_idade_paciente IS NULL) as missing_age,
            countIf(tipo_sexo_paciente IS NULL OR tipo_sexo_paciente = '') as missing_gender,
            countIf(sigla_uf_paciente IS NULL OR sigla_uf_paciente = '') as missing_state
        FROM events
        """

        result = client.query(general_query)
        general_stats = (
            dict(zip(result.column_names, result.result_rows[0]))
            if result.result_rows
            else {}
        )

        # Calculate data quality metrics
        total_records = general_stats.get("total_records", 0)
        data_quality = {
            "overall_completeness": round(
                (
                    1
                    - sum(
                        [
                            general_stats.get("missing_patient_id", 0),
                            general_stats.get("missing_date", 0),
                            general_stats.get("missing_vaccine", 0),
                        ]
                    )
                    / (total_records * 3)
                )
                * 100,
                2,
            )
            if total_records > 0
            else 0,
            "demographic_completeness": round(
                (
                    1
                    - sum(
                        [
                            general_stats.get("missing_age", 0),
                            general_stats.get("missing_gender", 0),
                            general_stats.get("missing_state", 0),
                        ]
                    )
                    / (total_records * 3)
                )
                * 100,
                2,
            )
            if total_records > 0
            else 0,
            "missing_rates": {
                "patient_id": round(
                    general_stats.get("missing_patient_id", 0) / total_records * 100, 2
                ),
                "vaccine": round(
                    general_stats.get("missing_vaccine", 0) / total_records * 100, 2
                ),
                "age": round(
                    general_stats.get("missing_age", 0) / total_records * 100, 2
                ),
                "gender": round(
                    general_stats.get("missing_gender", 0) / total_records * 100, 2
                ),
            }
            if total_records > 0
            else {},
        }

        return FilterStats(
            available_vaccines=available_vaccines,
            available_states=available_states,
            available_cities=available_cities,
            available_establishments=available_establishments,
            date_range={
                "min": str(general_stats.get("min_date", "")),
                "max": str(general_stats.get("max_date", "")),
            },
            age_range={
                "min": int(general_stats.get("min_age", 0))
                if general_stats.get("min_age") is not None
                else 0,
                "max": int(general_stats.get("max_age", 120))
                if general_stats.get("max_age") is not None
                else 120,
            },
            total_records=total_records,
            data_quality=data_quality,
        )

    except Exception as e:
        logger.error(f"Filter metadata error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metadata error: {str(e)}")


# =============================================================================
# ENHANCED PAGINATION WITH COMPREHENSIVE FILTERING
# =============================================================================


@router.post("/data/vaccinations", response_model=PaginationResponse)
@query_timer
async def get_vaccinations_with_comprehensive_filters(
    client: ClickHouseDep,
    filters: FilterRequest = FilterRequest(),
    pagination: PaginationRequest = PaginationRequest(),
) -> PaginationResponse:
    """Get paginated vaccination data with comprehensive filtering"""
    try:
        conditions, params = build_comprehensive_where_conditions(filters)

        # Add cursor-based pagination
        if pagination.cursor:
            try:
                cursor_value = int(pagination.cursor)
                conditions.append(
                    "row_number() OVER (ORDER BY data_vacina DESC) > %(cursor)s"
                )
                params["cursor"] = cursor_value
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid cursor format")

        params["limit"] = pagination.limit + 1  # Get one extra to check if there's more

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Validate sort_by field
        valid_sort_fields = [
            "data_vacina",
            "numero_idade_paciente",
            "sigla_vacina",
            "sigla_uf_paciente",
            "nome_municipio_paciente",
            "codigo_dose_vacina",
        ]

        if pagination.sort_by not in valid_sort_fields:
            pagination.sort_by = "data_vacina"

        query = f"""
        SELECT 
            row_number() OVER (ORDER BY {pagination.sort_by} {pagination.sort_order.value}) as row_id,
            codigo_paciente,
            sigla_vacina,
            descricao_vacina,
            data_vacina,
            numero_idade_paciente,
            tipo_sexo_paciente,
            nome_raca_cor_paciente,
            nome_municipio_paciente,
            sigla_uf_paciente,
            nome_uf_paciente,
            descricao_dose_vacina,
            codigo_dose_vacina,
            nome_razao_social_estabelecimento,
            descricao_tipo_estabelecimento,
            codigo_lote_vacina,
            descricao_local_aplicacao,
            codigo_cnes_estabelecimento
        FROM events 
        {where_clause}
        ORDER BY {pagination.sort_by} {pagination.sort_order.value}
        LIMIT %(limit)s
        """

        result = client.query(query, params)
        rows = result.result_rows
        columns = result.column_names

        data = [dict(zip(columns, row)) for row in rows]

        has_more = len(data) > pagination.limit
        if has_more:
            data = data[:-1]

        next_cursor = str(data[-1]["row_id"]) if data and has_more else None

        return PaginationResponse(
            data=data,
            pagination={
                "next_cursor": next_cursor,
                "has_more": has_more,
                "current_page_size": len(data),
                "requested_limit": pagination.limit,
            },
            metadata={
                "filters_applied": len(
                    [k for k, v in filters.model_dump(exclude_unset=True).items() if v]
                ),
                "sort_by": pagination.sort_by,
                "sort_order": pagination.sort_order.value,
                "data_freshness": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Vaccination data retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data retrieval error: {str(e)}")


# =============================================================================
# ADVANCED ANALYTICS ENDPOINTS
# =============================================================================


@router.post("/analytics/time-series")
@query_timer
async def get_advanced_time_series(
    client: ClickHouseDep,
    filters: FilterRequest = FilterRequest(),
    granularity: TimeGranularity = TimeGranularity.day,
    metrics: List[str] = Query(
        ["total_vaccinations", "unique_patients"], description="Metrics to include"
    ),
) -> ChartData:
    """Get advanced time series data with multiple metrics"""
    try:
        conditions, params = build_comprehensive_where_conditions(filters)

        # Default to last 90 days if no date range specified
        if not filters.start_date and not filters.end_date:
            conditions.append(get_clickhouse_date_condition(90))

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Choose date truncation based on granularity
        date_trunc_map = {
            TimeGranularity.day: "toDate(data_vacina)",
            TimeGranularity.week: "toStartOfWeek(data_vacina)",
            TimeGranularity.month: "toStartOfMonth(data_vacina)",
            TimeGranularity.quarter: "toStartOfQuarter(data_vacina)",
            TimeGranularity.year: "toStartOfYear(data_vacina)",
        }

        date_trunc = date_trunc_map[granularity]

        # Build metrics selection
        metrics_map = {
            "total_vaccinations": "count(*) as total_vaccinations",
            "unique_patients": "uniq(codigo_paciente) as unique_patients",
            "unique_vaccines": "uniq(sigla_vacina) as unique_vaccines",
            "avg_age": "round(avg(numero_idade_paciente), 2) as avg_age",
            "female_count": "countIf(tipo_sexo_paciente = 'F') as female_count",
            "male_count": "countIf(tipo_sexo_paciente = 'M') as male_count",
            "first_doses": "countIf(toInt32OrNull(codigo_dose_vacina) = 1) as first_doses",
            "second_doses": "countIf(toInt32OrNull(codigo_dose_vacina) = 2) as second_doses",
            "boosters": "countIf(toInt32OrNull(codigo_dose_vacina) >= 3) as boosters",
        }

        selected_metrics = [
            metrics_map.get(metric, metrics_map["total_vaccinations"])
            for metric in metrics
            if metric in metrics_map
        ]
        if not selected_metrics:
            selected_metrics = [metrics_map["total_vaccinations"]]

        query = f"""
        SELECT 
            {date_trunc} as period,
            {', '.join(selected_metrics)}
        FROM events 
        {where_clause}
        GROUP BY {date_trunc}
        ORDER BY period ASC
        LIMIT 1000
        """

        result = client.query(query, params)
        columns = result.column_names
        data = [dict(zip(columns, row)) for row in result.result_rows]

        return ChartData(
            chart_type=ChartType.line,
            title=f"Time Series Analysis ({granularity.value})",
            data=data,
            options={
                "responsive": True,
                "interaction": {"intersect": False},
                "scales": {"x": {"type": "time"}, "y": {"beginAtZero": True}},
            },
            metadata={
                "granularity": granularity.value,
                "metrics": metrics,
                "data_points": len(data),
            },
        )

    except Exception as e:
        logger.error(f"Time series error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Time series error: {str(e)}")


@router.post("/analytics/geographic-distribution")
@query_timer
async def get_geographic_distribution(
    client: ClickHouseDep,
    filters: FilterRequest = FilterRequest(),
    level: str = Query("state", pattern="^(state|city|region)$"),
    top_n: int = Query(20, ge=1, le=100),
) -> ChartData:
    """Get geographic distribution with multiple aggregation levels"""
    try:
        conditions, params = build_comprehensive_where_conditions(filters)
        params["top_n"] = top_n

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        if level == "state":
            query = f"""
            SELECT 
                sigla_uf_estabelecimento as code,
                nome_uf_estabelecimento as name,
                count(*) as total_vaccinations,
                uniq(codigo_paciente) as unique_patients,
                round(avg(numero_idade_paciente), 2) as avg_age,
                countIf(tipo_sexo_paciente = 'F') as female_count,
                countIf(tipo_sexo_paciente = 'M') as male_count,
                uniq(nome_municipio_estabelecimento) as cities_count,
                countIf(toInt32OrNull(codigo_dose_vacina) >= 2) as completed_series,
                round(countIf(toInt32OrNull(codigo_dose_vacina) >= 2) / uniq(codigo_paciente) * 100, 2) as completion_rate
            FROM events 
            {where_clause}
            GROUP BY sigla_uf_estabelecimento, nome_uf_estabelecimento
            ORDER BY total_vaccinations DESC
            LIMIT %(top_n)s
            """
        elif level == "city":
            query = f"""
            SELECT 
                nome_municipio_estabelecimento as name,
                sigla_uf_estabelecimento as state_code,
                nome_uf_estabelecimento as state_name,
                count(*) as total_vaccinations,
                uniq(codigo_paciente) as unique_patients,
                round(avg(numero_idade_paciente), 2) as avg_age
            FROM events 
            {where_clause}
            GROUP BY nome_municipio_estabelecimento, sigla_uf_estabelecimento, nome_uf_estabelecimento
            ORDER BY total_vaccinations DESC
            LIMIT %(top_n)s
            """
        else:  # region
            query = f"""
            SELECT 
                CASE 
                    WHEN sigla_uf_estabelecimento IN ('AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO') THEN 'Norte'
                    WHEN sigla_uf_estabelecimento IN ('AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE') THEN 'Nordeste'
                    WHEN sigla_uf_estabelecimento IN ('GO', 'MT', 'MS', 'DF') THEN 'Centro-Oeste'
                    WHEN sigla_uf_estabelecimento IN ('ES', 'MG', 'RJ', 'SP') THEN 'Sudeste'
                    WHEN sigla_uf_estabelecimento IN ('PR', 'RS', 'SC') THEN 'Sul'
                    ELSE 'Outros'
                END as region,
                count(*) as total_vaccinations,
                uniq(codigo_paciente) as unique_patients,
                round(avg(numero_idade_paciente), 2) as avg_age,
                uniq(sigla_uf_estabelecimento) as states_count,
                uniq(nome_municipio_estabelecimento) as cities_count
            FROM events 
            {where_clause}
            GROUP BY region
            ORDER BY total_vaccinations DESC
            """

        result = client.query(query, params)
        columns = result.column_names
        data = [dict(zip(columns, row)) for row in result.result_rows]

        return ChartData(
            chart_type=ChartType.bar,
            title=f"Geographic Distribution by {level.title()}",
            data=data,
            options={"responsive": True, "indexAxis": "y" if level == "city" else "x"},
            metadata={"level": level, "total_locations": len(data)},
        )

    except Exception as e:
        logger.error(f"Geographic distribution error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Geographic distribution error: {str(e)}"
        )


@router.post("/analytics/demographic-insights")
@query_timer
async def get_demographic_insights(
    client: ClickHouseDep,
    filters: FilterRequest = FilterRequest(),
    breakdown_by: str = Query(
        "age_gender", pattern="^(age_gender|age_only|gender_only|race)$"
    ),
) -> Dict[str, Any]:
    """Get comprehensive demographic insights with various breakdowns"""
    try:
        conditions, params = build_comprehensive_where_conditions(filters)
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        insights = {}

        if breakdown_by in ["age_gender", "age_only"]:
            # Age group analysis
            age_query = f"""
            SELECT 
                multiIf(
                    numero_idade_paciente < 18, '0-17',
                    numero_idade_paciente < 30, '18-29',
                    numero_idade_paciente < 50, '30-49',
                    numero_idade_paciente < 65, '50-64',
                    '65+'
                ) as age_group,
                {'' if breakdown_by == 'age_only' else 'tipo_sexo_paciente as gender,'}
                count(*) as count,
                uniq(codigo_paciente) as unique_patients,
                round(avg(numero_idade_paciente), 2) as avg_age
            FROM events 
            {where_clause}
            GROUP BY age_group{'' if breakdown_by == 'age_only' else ', gender'}
            ORDER BY age_group{'' if breakdown_by == 'age_only' else ', gender'}
            """

            result = client.query(age_query, params)
            columns = result.column_names
            age_data = [dict(zip(columns, row)) for row in result.result_rows]

            insights["age_distribution"] = age_data

        if breakdown_by in ["age_gender", "gender_only"]:
            # Gender analysis
            gender_query = f"""
            SELECT 
                tipo_sexo_paciente as gender,
                count(*) as count,
                uniq(codigo_paciente) as unique_patients,
                round(avg(numero_idade_paciente), 2) as avg_age,
                -- Age group breakdown within gender
                countIf(numero_idade_paciente < 18) as children,
                countIf(numero_idade_paciente >= 18 AND numero_idade_paciente < 65) as adults,
                countIf(numero_idade_paciente >= 65) as seniors
            FROM events 
            {where_clause}
            GROUP BY tipo_sexo_paciente
            ORDER BY count DESC
            """

            result = client.query(gender_query, params)
            columns = result.column_names
            gender_data = [dict(zip(columns, row)) for row in result.result_rows]

            insights["gender_distribution"] = gender_data

        if breakdown_by == "race":
            # Race/ethnicity analysis
            race_query = f"""
            SELECT 
                nome_raca_cor_paciente as race_color,
                count(*) as count,
                uniq(codigo_paciente) as unique_patients,
                round(avg(numero_idade_paciente), 2) as avg_age,
                countIf(tipo_sexo_paciente = 'F') as female_count,
                countIf(tipo_sexo_paciente = 'M') as male_count
            FROM events 
            {where_clause}
            AND nome_raca_cor_paciente IS NOT NULL 
            AND nome_raca_cor_paciente != ''
            GROUP BY nome_raca_cor_paciente
            ORDER BY count DESC
            LIMIT 20
            """

            result = client.query(race_query, params)
            columns = result.column_names
            race_data = [dict(zip(columns, row)) for row in result.result_rows]

            insights["race_distribution"] = race_data

        # Summary statistics
        summary_query = f"""
        SELECT 
            count(*) as total_vaccinations,
            uniq(codigo_paciente) as total_patients,
            round(avg(numero_idade_paciente), 2) as overall_avg_age,
            min(numero_idade_paciente) as min_age,
            max(numero_idade_paciente) as max_age,
            countIf(tipo_sexo_paciente = 'F') as total_female,
            countIf(tipo_sexo_paciente = 'M') as total_male
        FROM events 
        {where_clause}
        """

        result = client.query(summary_query, params)
        summary_data = (
            dict(zip(result.column_names, result.result_rows[0]))
            if result.result_rows
            else {}
        )

        insights["summary"] = summary_data
        insights["metadata"] = {
            "breakdown_by": breakdown_by,
            "generated_at": datetime.now().isoformat(),
            "filters_applied": len(
                [k for k, v in filters.model_dump(exclude_unset=True).items() if v]
            ),
        }

        return insights

    except Exception as e:
        logger.error(f"Demographic insights error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Demographic insights error: {str(e)}"
        )


@router.post("/analytics/vaccine-performance")
@query_timer
async def get_vaccine_performance_analysis(
    client: ClickHouseDep,
    filters: FilterRequest = FilterRequest(),
    include_trends: bool = True,
) -> Dict[str, Any]:
    """Comprehensive vaccine performance analysis"""
    try:
        conditions, params = build_comprehensive_where_conditions(filters)
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Main vaccine performance metrics
        performance_query = f"""
        SELECT 
            sigla_vacina as vaccine_code,
            descricao_vacina as vaccine_name,
            count(*) as total_doses,
            uniq(codigo_paciente) as unique_patients,
            round(avg(numero_idade_paciente), 2) as avg_patient_age,
            min(data_vacina) as first_dose_date,
            max(data_vacina) as last_dose_date,
            uniq(sigla_uf_paciente) as states_covered,
            uniq(nome_municipio_paciente) as cities_covered,
            countIf(tipo_sexo_paciente = 'F') as female_patients,
            countIf(tipo_sexo_paciente = 'M') as male_patients,
            countIf(toInt32OrNull(codigo_dose_vacina)  = 1) as first_doses,
            countIf(toInt32OrNull(codigo_dose_vacina)  = 2) as second_doses,
            countIf(toInt32OrNull(codigo_dose_vacina)  >= 3) as boosters,
            round(countIf(toInt32OrNull(codigo_dose_vacina)  >= 2) / uniq(codigo_paciente) * 100, 2) as completion_rate
        FROM events 
        {where_clause}
        GROUP BY sigla_vacina, descricao_vacina
        ORDER BY total_doses DESC
        """

        result = client.query(performance_query, params)
        columns = result.column_names
        vaccine_performance = [dict(zip(columns, row)) for row in result.result_rows]

        # Calculate derived metrics
        for vaccine in vaccine_performance:
            vaccine["doses_per_patient"] = (
                round(vaccine["total_doses"] / vaccine["unique_patients"], 2)
                if vaccine["unique_patients"] > 0
                else 0
            )

        analysis = {
            "vaccine_performance": vaccine_performance,
            "summary": {
                "total_vaccines": len(vaccine_performance),
                "total_doses": sum(v["total_doses"] for v in vaccine_performance),
                "total_patients": sum(
                    v["unique_patients"] for v in vaccine_performance
                ),
                "avg_completion_rate": round(
                    sum(v["completion_rate"] for v in vaccine_performance)
                    / len(vaccine_performance),
                    2,
                )
                if vaccine_performance
                else 0,
            },
        }

        if include_trends:
            # Vaccine trends over time (last 90 days)
            trend_conditions = conditions.copy()
            trend_conditions.append(get_clickhouse_date_condition(90))
            trend_where_clause = (
                "WHERE " + " AND ".join(trend_conditions) if trend_conditions else ""
            )

            trends_query = f"""
            SELECT 
                sigla_vacina as vaccine_code,
                toDate(data_vacina) as date,
                count(*) as daily_doses,
                uniq(codigo_paciente) as daily_patients
            FROM events 
            {trend_where_clause}
            GROUP BY sigla_vacina, toDate(data_vacina)
            ORDER BY vaccine_code, date
            """

            result = client.query(trends_query, params)
            trends_data = [
                dict(zip(result.column_names, row)) for row in result.result_rows
            ]

            # Group trends by vaccine
            trends_by_vaccine = {}
            for row in trends_data:
                vaccine = row["vaccine_code"]
                if vaccine not in trends_by_vaccine:
                    trends_by_vaccine[vaccine] = []
                trends_by_vaccine[vaccine].append(
                    {
                        "date": str(row["date"]),
                        "daily_doses": row["daily_doses"],
                        "daily_patients": row["daily_patients"],
                    }
                )

            analysis["trends"] = trends_by_vaccine

        analysis["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "include_trends": include_trends,
            "filters_applied": len(
                [k for k, v in filters.dict(exclude_unset=True).items() if v]
            ),
        }

        return analysis

    except Exception as e:
        logger.error(f"Vaccine performance analysis error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Vaccine performance analysis error: {str(e)}"
        )


# =============================================================================
# ADVANCED SEARCH AND EXPORT
# =============================================================================


@router.post("/search/advanced")
@query_timer
async def advanced_search_with_filters(
    client: ClickHouseDep,
    filters: FilterRequest = FilterRequest(),
    search_terms: Optional[str] = Query(None, description="Search terms"),
    pagination: PaginationRequest = PaginationRequest(),
    export_format: Optional[str] = Query(None, pattern="^(csv|json)$"),
) -> Union[PaginationResponse, Dict[str, Any]]:
    """Advanced search with full-text capabilities and export options"""
    try:
        conditions, params = build_comprehensive_where_conditions(filters)

        # Add search conditions
        if search_terms:
            search_conditions = []
            search_words = search_terms.split()

            for i, word in enumerate(search_words[:5]):  # Limit to 5 search terms
                search_conditions.append(
                    f"""(
                    nome_razao_social_estabelecimento ILIKE %(search_{i})s OR
                    nome_municipio_paciente ILIKE %(search_{i})s OR
                    descricao_vacina ILIKE %(search_{i})s OR
                    codigo_lote_vacina ILIKE %(search_{i})s
                )"""
                )
                params[f"search_{i}"] = f"%{word}%"

            if search_conditions:
                conditions.extend(search_conditions)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Handle export format
        if export_format:
            export_query = f"""
            SELECT 
                codigo_paciente,
                sigla_vacina,
                descricao_vacina,
                data_vacina,
                numero_idade_paciente,
                tipo_sexo_paciente,
                nome_municipio_paciente,
                sigla_uf_paciente,
                descricao_dose_vacina,
                codigo_dose_vacina,
                nome_razao_social_estabelecimento,
                descricao_tipo_estabelecimento
            FROM events 
            {where_clause}
            ORDER BY data_vacina DESC
            LIMIT 1000
            """

            result = client.query(export_query, params)
            columns = result.column_names
            data = [dict(zip(columns, row)) for row in result.result_rows]

            return {
                "export_format": export_format,
                "total_records": len(data),
                "data": data,
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "search_terms": search_terms,
                },
            }

        # Regular search with pagination
        if pagination.cursor:
            conditions.append(
                "row_number() OVER (ORDER BY data_vacina DESC) > %(cursor)s"
            )
            params["cursor"] = int(pagination.cursor)

        params["limit"] = pagination.limit + 1

        search_query = f"""
        SELECT 
            row_number() OVER (ORDER BY data_vacina DESC) as row_id,
            codigo_paciente,
            sigla_vacina,
            descricao_vacina,
            data_vacina,
            numero_idade_paciente,
            tipo_sexo_paciente,
            nome_municipio_paciente,
            sigla_uf_paciente,
            descricao_dose_vacina,
            codigo_dose_vacina,
            nome_razao_social_estabelecimento
        FROM events 
        {where_clause}
        ORDER BY data_vacina DESC
        LIMIT %(limit)s
        """

        result = client.query(search_query, params)
        rows = result.result_rows
        columns = result.column_names

        data = [dict(zip(columns, row)) for row in rows]

        has_more = len(data) > pagination.limit
        if has_more:
            data = data[:-1]

        next_cursor = str(data[-1]["row_id"]) if data and has_more else None

        return PaginationResponse(
            data=data,
            pagination={
                "next_cursor": next_cursor,
                "has_more": has_more,
                "current_page_size": len(data),
            },
            metadata={
                "search_terms": search_terms,
                "search_performed": bool(search_terms),
            },
        )

    except Exception as e:
        logger.error(f"Advanced search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# =============================================================================
# REAL-TIME MONITORING
# =============================================================================


@router.get("/monitoring/real-time-stats")
@query_timer
async def get_real_time_statistics(client: ClickHouseDep) -> Dict[str, Any]:
    """Get real-time statistics for monitoring dashboard"""
    try:
        current_time = datetime.now()

        # Today's statistics
        today_query = f"""
        SELECT 
            count(*) as today_vaccinations,
            uniq(codigo_paciente) as today_patients,
            uniq(sigla_vacina) as vaccines_used_today,
            countIf(codigo_dose_vacina = 1) as today_first_doses,
            countIf(codigo_dose_vacina = 2) as today_second_doses,
            countIf(codigo_dose_vacina >= 3) as today_boosters
        FROM events
        WHERE {get_clickhouse_today_condition()}
        """

        result = client.query(today_query)
        today_stats = (
            dict(zip(result.column_names, result.result_rows[0]))
            if result.result_rows
            else {}
        )

        # Weekly trend
        weekly_trend_query = f"""
        SELECT 
            toDate(data_vacina) as date,
            count(*) as daily_count
        FROM events
        WHERE {get_clickhouse_date_condition(7)}
        GROUP BY toDate(data_vacina)
        ORDER BY date
        """

        result = client.query(weekly_trend_query)
        weekly_trend = [
            {"date": str(row[0]), "count": row[1]} for row in result.result_rows
        ]

        return {
            "timestamp": current_time.isoformat(),
            "today_stats": today_stats,
            "weekly_trend": weekly_trend,
            "system_health": {
                "status": "healthy",
                "last_update": current_time.isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"Real-time statistics error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Real-time statistics error: {str(e)}"
        )


# =============================================================================
# HEALTH CHECK
# =============================================================================


@router.get("/health")
async def health_check(client: ClickHouseDep) -> Dict[str, Any]:
    """Comprehensive health check"""
    try:
        start_time = time.time()
        result = client.query("SELECT count(*) FROM events LIMIT 1")
        db_response_time = (time.time() - start_time) * 1000

        total_records = result.result_rows[0][0] if result.result_rows else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "api_version": "3.0.0",
            "status": "healthy",
            "database": {
                "status": "connected",
                "response_time_ms": round(db_response_time, 2),
                "total_records": total_records,
            },
        }

    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "unhealthy",
            "error": str(e),
        }
