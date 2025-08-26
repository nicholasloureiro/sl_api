SEMANTIC_MODEL = {
    "tables": [
        {
            "table_name": "vacinas.events",
            "table_description": (
                "Individual vaccination event records from Brazil’s RNDS feed. Includes patient demographics "
                "(sex, race/color, nationality, state), immunobiologic details (vaccine code/name, brand, dose, route), "
                "timestamps (vaccination date, RNDS ingestion date), administering facility data (legal name, city, state, ownership), "
                "and campaign/strategy metadata. Example columns: descricao_vacina, codigo_vacina, sigla_vacina, "
                "codigo_dose_vacina, descricao_dose_vacina, descricao_via_administracao, codigo_via_administracao, "
                "nome_raca_cor_paciente, tipo_sexo_paciente, sigla_uf_paciente, nome_municipio_estabelecimento, "
                "sigla_uf_estabelecimento, nome_razao_social_estabelecimento, data_vacina, data_entrada_rnds."
            ),
            "use_case": (
                "Coverage and adherence analysis over time; breakdowns by region (state/municipality), facility, and demographics; "
                "time series by immunobiologic and dose (first, second, booster); campaign/strategy effectiveness; "
                "manufacturer comparisons; detection of delays and dose abandonment patterns; "
                "operational dashboards for daily/weekly/monthly applications."
            ),
        }
    ]
}
