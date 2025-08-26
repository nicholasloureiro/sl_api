clickhouse_rules = """
# ClickHouse Rules and Best Practices for Vaccination Analytics

## Query Rules
- Always use LIMIT instead of TOP for row limiting
- Use lowercase table and column names consistently (e.g., vacinas.events, data_vacina)
- Prefer ClickHouse functions: length(), substring(), position(), now(), toDate(), toStartOfMonth(), dateDiff(), dateAdd()
- Handle NULL values explicitly using functions like ifNull(), assumeNotNull(), toInt32OrNull()
- Use appropriate JOINs (INNER, LEFT) but note ClickHouse joins can be memory intensive — aggregate before joining when possible
- When aggregating, use uniq() or uniqExact() for distinct counts instead of COUNT(DISTINCT …)

## Performance Guidelines
- Avoid SELECT * in analytical queries — only select needed columns
- Use WHERE clauses with partitioning columns (e.g., data_vacina) to minimize scanned data
- Leverage materialized views or pre-aggregated tables for heavy recurring queries
- Consider using APPROXIMATE functions (e.g., uniq, quantile) for large-scale analytics
- Use WITH clauses (CTEs) to simplify complex queries, but remember they are inlined in ClickHouse

## Security Guidelines
- Do not allow DELETE, UPDATE, INSERT, or DROP operations in analytical queries
- Always validate and sanitize filter parameters (e.g., state codes, vaccine names)
- Use parameterized queries for dashboards and APIs
- Limit result sets with LIMIT and aggregation to prevent huge payloads

## Vaccination Data Specific Rules
- Patient information is anonymized; analyses must focus on aggregate trends
- Use data_vacina for time-series analysis (prefer toDate() or time-bucket functions like toStartOfMonth)
- Dose completion should be calculated via codigo_dose_vacina (1 = first, 2 = second, >=3 = boosters)
- Geographic analysis should use sigla_uf_paciente (state), nome_municipio_paciente (city), and regional groupings
- Establishment performance is tracked via codigo_cnes_estabelecimento and nome_razao_social_estabelecimento
- Lot-level monitoring is available through codigo_lote_vacina
"""
