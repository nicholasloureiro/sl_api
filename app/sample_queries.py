# sample_queries.py

sample_queries = """
# Vaccination Sample Queries

## Coverage Analysis

### Daily Vaccination Trends (Last 90 Days)
```sql
SELECT
    toDate(data_vacina) AS Day,
    count(*) AS TotalDoses,
    uniq(codigo_paciente) AS UniquePatients
FROM vacinas.events
WHERE toDate(data_vacina) >= toDate(now()) - INTERVAL 90 DAY
GROUP BY Day
ORDER BY Day
```

### Monthly Coverage by Dose
```sql
SELECT
    toStartOfMonth(data_vacina) AS Month,
    countIf(toInt32OrNull(codigo_dose_vacina) = 1) AS FirstDoses,
    countIf(toInt32OrNull(codigo_dose_vacina) = 2) AS SecondDoses,
    countIf(toInt32OrNull(codigo_dose_vacina) >= 3) AS Boosters
FROM vacinas.events
GROUP BY Month
ORDER BY Month
```

### Year-to-Date Coverage and Completion Rate
```sql
WITH ytd AS (
  SELECT
    count(*) AS TotalDoses,
    uniq(codigo_paciente) AS UniquePatients,
    countIf(toInt32OrNull(codigo_dose_vacina) >= 2) AS CompletedSeries
  FROM vacinas.events
  WHERE toStartOfYear(data_vacina) = toStartOfYear(today())
)
SELECT
  TotalDoses,
  UniquePatients,
  CompletedSeries,
  roundIf(CompletedSeries / UniquePatients * 100, 2, UniquePatients > 0) AS CompletionRatePct
FROM ytd
```

### Rolling 7-Day Average (Doses)
```sql
SELECT
  Day,
  DailyDoses,
  avgOver(DailyDoses) OVER (ORDER BY Day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS Rolling7dAvg
FROM (
  SELECT toDate(data_vacina) AS Day, count(*) AS DailyDoses
  FROM vacinas.events
  GROUP BY Day
)
ORDER BY Day
```

## Geographic Analysis

### Top 20 States by Total Doses
```sql
SELECT
    sigla_uf_paciente AS StateCode,
    anyLast(nome_uf_paciente) AS StateName,
    count(*) AS TotalDoses,
    uniq(codigo_paciente) AS UniquePatients,
    roundIf(countIf(toInt32OrNull(codigo_dose_vacina) >= 2) / uniq(codigo_paciente) * 100, 2, uniq(codigo_paciente) > 0) AS CompletionRatePct
FROM vacinas.events
GROUP BY sigla_uf_paciente
ORDER BY TotalDoses DESC
LIMIT 20
```

### Top 20 Cities in São Paulo by Doses
```sql
SELECT
    nome_municipio_paciente AS City,
    sigla_uf_paciente AS StateCode,
    count(*) AS TotalDoses,
    uniq(codigo_paciente) AS UniquePatients
FROM vacinas.events
WHERE sigla_uf_paciente = 'SP'
GROUP BY City, StateCode
ORDER BY TotalDoses DESC
LIMIT 20
```

### Regional Distribution (Brazilian Macro-Regions)
```sql
SELECT 
  CASE 
    WHEN sigla_uf_paciente IN ('AC','AP','AM','PA','RO','RR','TO') THEN 'Norte'
    WHEN sigla_uf_paciente IN ('AL','BA','CE','MA','PB','PE','PI','RN','SE') THEN 'Nordeste'
    WHEN sigla_uf_paciente IN ('GO','MT','MS','DF') THEN 'Centro-Oeste'
    WHEN sigla_uf_paciente IN ('ES','MG','RJ','SP') THEN 'Sudeste'
    WHEN sigla_uf_paciente IN ('PR','RS','SC') THEN 'Sul'
    ELSE 'Other'
  END AS Region,
  count(*) AS TotalDoses,
  uniq(codigo_paciente) AS UniquePatients,
  round(avg(numero_idade_paciente), 2) AS AvgAge,
  uniq(sigla_uf_paciente) AS StatesCount,
  uniq(nome_municipio_paciente) AS CitiesCount
FROM vacinas.events
GROUP BY Region
ORDER BY TotalDoses DESC
```

## Vaccine Analysis

### Top Vaccines by Doses and Patients
```sql
SELECT
    sigla_vacina AS VaccineCode,
    anyLast(descricao_vacina) AS VaccineName,
    count(*) AS TotalDoses,
    uniq(codigo_paciente) AS UniquePatients,
    roundIf(count(*) / uniq(codigo_paciente), 2, uniq(codigo_paciente) > 0) AS DosesPerPatient
FROM vacinas.events
GROUP BY sigla_vacina
ORDER BY TotalDoses DESC
LIMIT 15
```

### Dose Mix by Vaccine (Share %)
```sql
SELECT
  sigla_vacina AS VaccineCode,
  anyLast(descricao_vacina) AS VaccineName,
  countIf(toInt32OrNull(codigo_dose_vacina) = 1) AS FirstDoses,
  countIf(toInt32OrNull(codigo_dose_vacina) = 2) AS SecondDoses,
  countIf(toInt32OrNull(codigo_dose_vacina) >= 3) AS Boosters,
  round(FirstDoses * 100.0 / count(*), 2) AS PctFirst,
  round(SecondDoses * 100.0 / count(*), 2) AS PctSecond,
  round(Boosters * 100.0 / count(*), 2) AS PctBoosters
FROM vacinas.events
GROUP BY sigla_vacina
ORDER BY count(*) DESC
LIMIT 15
```

### Vaccine Trends Over Time (Last 90 Days)
```sql
SELECT 
  sigla_vacina AS VaccineCode,
  toDate(data_vacina) AS Day,
  count(*) AS DailyDoses,
  uniq(codigo_paciente) AS DailyPatients
FROM vacinas.events
WHERE toDate(data_vacina) >= toDate(now()) - INTERVAL 90 DAY
GROUP BY VaccineCode, Day
ORDER BY VaccineCode, Day
```

## Demographics

### Age Buckets Distribution
```sql
SELECT
  multiIf(
    numero_idade_paciente < 18, '0-17',
    numero_idade_paciente < 30, '18-29',
    numero_idade_paciente < 50, '30-49',
    numero_idade_paciente < 65, '50-64',
    '65+'
  ) AS AgeGroup,
  count(*) AS TotalDoses,
  uniq(codigo_paciente) AS UniquePatients
FROM vacinas.events
GROUP BY AgeGroup
ORDER BY arrayPosition(['0-17','18-29','30-49','50-64','65+'], AgeGroup)
```

### Gender Split per Top 10 Vaccines
```sql
WITH TopVac AS (
  SELECT sigla_vacina
  FROM vacinas.events
  GROUP BY sigla_vacina
  ORDER BY count(*) DESC
  LIMIT 10
)
SELECT
  e.sigla_vacina AS VaccineCode,
  e.tipo_sexo_paciente AS Gender,
  count(*) AS TotalDoses,
  uniq(e.codigo_paciente) AS UniquePatients
FROM vacinas.events e
INNER JOIN TopVac t ON e.sigla_vacina = t.sigla_vacina
GROUP BY VaccineCode, Gender
ORDER BY VaccineCode, TotalDoses DESC
```

### Race/Color Distribution (Top 20)
```sql
SELECT 
  nome_raca_cor_paciente AS RaceColor,
  count(*) AS TotalDoses,
  uniq(codigo_paciente) AS UniquePatients,
  round(avg(numero_idade_paciente), 2) AS AvgAge,
  countIf(tipo_sexo_paciente = 'F') AS FemaleCount,
  countIf(tipo_sexo_paciente = 'M') AS MaleCount
FROM vacinas.events
WHERE nome_raca_cor_paciente IS NOT NULL AND nome_raca_cor_paciente != ''
GROUP BY RaceColor
ORDER BY TotalDoses DESC
LIMIT 20
```

## Establishments (CNES)

### Top Establishments by Doses (with Type)
```sql
SELECT
  codigo_cnes_estabelecimento AS CNESCode,
  anyLast(nome_razao_social_estabelecimento) AS EstablishmentName,
  anyLast(descricao_tipo_estabelecimento) AS EstablishmentType,
  sigla_uf_paciente AS StateCode,
  count(*) AS TotalDoses,
  uniq(codigo_paciente) AS UniquePatients
FROM vacinas.events
GROUP BY CNESCode, StateCode
ORDER BY TotalDoses DESC
LIMIT 25
```

### City-Level Establishment Leaders (São Paulo Example)
```sql
SELECT
  anyLast(nome_municipio_paciente) AS City,
  anyLast(sigla_uf_paciente) AS StateCode,
  codigo_cnes_estabelecimento AS CNESCode,
  anyLast(nome_razao_social_estabelecimento) AS EstablishmentName,
  count(*) AS TotalDoses
FROM vacinas.events
WHERE nome_municipio_paciente = 'São Paulo' AND sigla_uf_paciente = 'SP'
GROUP BY CNESCode
ORDER BY TotalDoses DESC
LIMIT 20
```

## Series Completion & Conversion

### Completion Rate by State (2+ Doses over Unique Patients)
```sql
SELECT
  sigla_uf_paciente AS StateCode,
  uniq(codigo_paciente) AS UniquePatients,
  countIf(toInt32OrNull(codigo_dose_vacina) >= 2) AS CompletedSeries,
  roundIf(CompletedSeries / UniquePatients * 100, 2, UniquePatients > 0) AS CompletionRatePct
FROM vacinas.events
GROUP BY StateCode
ORDER BY CompletionRatePct DESC
```

### First-to-Second Dose Conversion (Cohort within 120 Days)
```sql
WITH first_dose AS (
  SELECT
    codigo_paciente,
    minIf(toDate(data_vacina), toInt32OrNull(codigo_dose_vacina) = 1) AS FirstDate
  FROM vacinas.events
  GROUP BY codigo_paciente
),
second_dose AS (
  SELECT
    codigo_paciente,
    minIf(toDate(data_vacina), toInt32OrNull(codigo_dose_vacina) = 2) AS SecondDate
  FROM vacinas.events
  GROUP BY codigo_paciente
)
SELECT
  count() AS PatientsWithFirst,
  countIf(SecondDate IS NOT NULL AND SecondDate <= FirstDate + 120) AS Converted120d,
  roundIf(Converted120d / PatientsWithFirst * 100, 2, PatientsWithFirst > 0) AS ConversionRate120dPct
FROM first_dose f
LEFT JOIN second_dose s USING (codigo_paciente)
```

## Quality & Anomalies

### Lots with Unusual Concentration (Top 15)
```sql
SELECT
  codigo_lote_vacina AS LotCode,
  sigla_vacina AS VaccineCode,
  count(*) AS TotalDoses,
  uniq(codigo_paciente) AS UniquePatients,
  uniq(sigla_uf_paciente) AS StatesCount,
  uniq(codigo_cnes_estabelecimento) AS EstablishmentsCount
FROM vacinas.events
WHERE codigo_lote_vacina != '' AND codigo_lote_vacina IS NOT NULL
GROUP BY LotCode, VaccineCode
ORDER BY TotalDoses DESC
LIMIT 15
```

### Data Completeness Snapshot
```sql
SELECT
  count(*) AS TotalRecords,
  countIf(codigo_paciente = '' OR codigo_paciente IS NULL) AS MissingPatientId,
  countIf(sigla_vacina = '' OR sigla_vacina IS NULL) AS MissingVaccine,
  countIf(numero_idade_paciente IS NULL) AS MissingAge,
  countIf(tipo_sexo_paciente = '' OR tipo_sexo_paciente IS NULL) AS MissingGender,
  countIf(sigla_uf_paciente = '' OR sigla_uf_paciente IS NULL) AS MissingState,
  roundIf(MissingPatientId * 100.0 / TotalRecords, 2, TotalRecords > 0) AS PctMissingPatientId,
  roundIf(MissingVaccine * 100.0 / TotalRecords, 2, TotalRecords > 0) AS PctMissingVaccine
FROM vacinas.events
```

## Recent Activity & Monitoring

### Today vs Yesterday (Doses & Patients)
```sql
WITH today AS (
  SELECT count(*) AS Doses, uniq(codigo_paciente) AS Patients
  FROM vacinas.events
  WHERE toDate(data_vacina) = toDate(now())
),
yday AS (
  SELECT count(*) AS Doses, uniq(codigo_paciente) AS Patients
  FROM vacinas.events
  WHERE toDate(data_vacina) = toDate(now()) - 1
)
SELECT
  today.Doses AS TodayDoses,
  yday.Doses AS YesterdayDoses,
  today.Patients AS TodayPatients,
  yday.Patients AS YesterdayPatients,
  roundIf((today.Doses - yday.Doses) * 100.0 / nullIf(yday.Doses, 0), 2, yday.Doses > 0) AS DosesDeltaPct,
  roundIf((today.Patients - yday.Patients) * 100.0 / nullIf(yday.Patients, 0), 2, yday.Patients > 0) AS PatientsDeltaPct
FROM today, yday
```

### Top 10 Establishments Today
```sql
SELECT
  codigo_cnes_estabelecimento AS CNESCode,
  anyLast(nome_razao_social_estabelecimento) AS EstablishmentName,
  count(*) AS TodayDoses
FROM vacinas.events
WHERE toDate(data_vacina) = toDate(now())
GROUP BY CNESCode
ORDER BY TodayDoses DESC
LIMIT 10
```

## Ad-Hoc Filtering Examples

### Filter: Specific Vaccine + State + Date Range
```sql
SELECT
  toDate(data_vacina) AS Day,
  sigla_vacina AS VaccineCode,
  sigla_uf_paciente AS StateCode,
  count(*) AS TotalDoses,
  uniq(codigo_paciente) AS UniquePatients
FROM vacinas.events
WHERE sigla_vacina = 'BCG'
  AND sigla_uf_paciente = 'SP'
  AND toDate(data_vacina) BETWEEN toDate('2025-01-01') AND toDate('2025-06-30')
GROUP BY Day, VaccineCode, StateCode
ORDER BY Day
```
"""