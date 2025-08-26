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

## Domain-Specific Rules
- In `vacinas.events`, the column `nome_raca_cor_paciente` has **fixed categories**:
    - PARDA
    - BRANCA
    - AMARELA
    - INDIGENA
    - PRETA
    - SEM INFORMACAO

  ⚠️ Important: values are always ALL CAPS and WITHOUT ACCENTS. 

## Normalização de Categoria (raça/cor)
- Em `vacinas.events`, `nome_raca_cor_paciente` tem valores FIXOS (ALL CAPS, sem acentos):
  PARDA, BRANCA, AMARELA, INDIGENA, PRETA, SEM INFORMACAO
- Se o usuário escrever "indígena" (com acento/minúsculas), mapear para 'INDIGENA'.

## Datas & Idades
- Idade: `dateDiff('year', data_nascimento_paciente, today())`.
- Para deduplicar por paciente, agregue nascimento com `min(data_nascimento_paciente)`.
- "Idoso" = idade >= 60 (padrão Brasil), salvo se o usuário definir outro critério.

## Desambiguação de intenção → padrões SQL
- "mais tomou vacinas" → agrupar por `codigo_paciente` e ordenar por `count() DESC`.
- "mais velho" → ordenar por nascimento ASC (ou idade DESC), após filtrar idade.

## Exemplos prontos

-- 1) "Qual o indígena que mais tomou vacinas da base?"
SELECT
  codigo_paciente,
  count() AS total_doses
FROM vacinas.events
WHERE nome_raca_cor_paciente = 'INDIGENA'
GROUP BY codigo_paciente
ORDER BY total_doses DESC
LIMIT 1;

-- 2) "Qual o idoso mais velho da base?"
-- Define-se idoso como idade >= 60 (ajuste se o usuário pedir outro limiar).
SELECT
  codigo_paciente,
  min(data_nascimento_paciente) AS nascimento,
  dateDiff('year', min(data_nascimento_paciente), today()) AS idade
FROM vacinas.events
WHERE data_nascimento_paciente IS NOT NULL
GROUP BY codigo_paciente
HAVING idade >= 60
ORDER BY nascimento ASC
LIMIT 1;

## Observações
- Se houver múltiplos empates (mesma contagem/mesma idade), use um ORDER BY secundário
  estável, por ex.: ORDER BY total_doses DESC, codigo_paciente ASC.
- Ajuste o nome da coluna de nascimento se o schema diferir; o padrão esperado é
  `data_nascimento_paciente`.
"""
