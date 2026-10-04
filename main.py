import pandas as pd
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS


FILE = "Sample_data.xlsx"

VARIABLES = {
    "2012-2022": [
        "Suskil", "Futsup", "Knoent",
        "Frfail", "Opport", "RQ"
    ],
    "2019-2022": [
        "Suskil", "Futsup", "Knoent",
        "Frfail", "Opport", "Nbstat", "RQ"
    ],
}


def read_sample(sheet_name, variables):
    df = pd.read_excel(FILE, sheet_name=sheet_name)

    df = df.rename(columns={
        "Country": "country",
        "Year": "year",
        "Region": "region",
        "Income Group": "income_group",
        "TEA_18_34": "tea",
    })

    needed = [
        "country", "year", "region", "income_group",
        "tea", *variables
    ]
    missing = [col for col in needed if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {sheet_name}: {missing}"
        )

    for col in ["tea", *variables]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country", "year"]).copy()
    df["year"] = df["year"].astype(int)

    if df.duplicated(["country", "year"]).any():
        raise ValueError(
            f"{sheet_name} has duplicate country-year rows."
        )

    return df


def prepare_model_data(df, variables, sheet_name):
    data = df.dropna(
        subset=["tea", "region", "income_group", *variables]
    ).copy()

    panel = data.set_index(["country", "year"]).sort_index()
    y = panel["tea"].astype(float)

    controls = pd.get_dummies(
        panel[["region", "income_group"]],
        drop_first=True,
        dtype=float,
    )
    # In both 2012–2022 sheets, this dummy duplicates the
    # lower-middle-income dummy.
    if sheet_name.startswith("S2012-2022"):
        controls = controls.drop(columns="region_South Asia")

    x_pooled_re = pd.concat(
        [panel[variables].astype(float), controls],
        axis=1,
    )
    x_pooled_re["constant"] = 1.0

    # Country fixed effects absorb region and income group.
    x_fe = panel[variables].astype(float)

    countries = data["country"].nunique()

    return y, x_pooled_re, x_fe, countries


def make_column(result, included_vars, all_vars, countries, fe=False):
    column = {}

    for var in all_vars:
        if var in included_vars:
            column[var] = result.params[var]
            column[f"{var} SE"] = result.std_errors[var]
        else:
            column[var] = float("nan")
            column[f"{var} SE"] = float("nan")

    column["N"] = result.nobs
    column["Countries"] = countries

    if fe:
        column["Within R²"] = result.rsquared_within
    else:
        column["R²"] = result.rsquared

    return column


def run_sample(sheet_name, variables):
    df = read_sample(sheet_name, variables)

    # Joint model: pooled OLS, random effects, country fixed effects.
    y, x_pooled_re, x_fe, countries = prepare_model_data(
        df, variables, sheet_name
    )

    pooled = PooledOLS(y, x_pooled_re).fit(
        cov_type="clustered",
        cluster_entity=True,
    )

    random = RandomEffects(y, x_pooled_re).fit(
        cov_type="clustered",
        cluster_entity=True,
    )
    
    # print(f"\nTHETA — {sheet_name}")
    # print(random.theta.round(4).to_string())

    fixed = PanelOLS(
        y,
        x_fe,
        entity_effects=True,
        time_effects=True,
    ).fit(
        cov_type="clustered",
        cluster_entity=True,
    )

    # Display explanatory variables and control dummies.
    displayed_vars = [
        col for col in x_pooled_re.columns
        if col != "constant"
    ]

    # Country FE: each variable alone, then the joint model.
    fe_columns = {}

    for number, var in enumerate(variables, start=1):
        single_y, _, single_x_fe, single_countries = (
            prepare_model_data(df, [var], sheet_name)
        )

        single_fe = PanelOLS(
            single_y,
            single_x_fe,
            entity_effects=True,
            time_effects=True,
        ).fit(
            cov_type="clustered",
            cluster_entity=True,
        )

        fe_columns[f"({number}) {var}"] = make_column(
            single_fe,
            included_vars=[var],
            all_vars=variables,
            countries=single_countries,
            fe=True,
        )

    fe_columns[f"({len(variables) + 1}) Joint"] = make_column(
        fixed,
        included_vars=variables,
        all_vars=variables,
        countries=countries,
        fe=True,
    )

    fe_table = pd.DataFrame(fe_columns)

    univariate_ols_columns = {}
    univariate_re_columns = {}

    for number, var in enumerate(variables, start=1):
        single_data = df.dropna(subset=["tea", var]).copy()
        single_panel = single_data.set_index(
            ["country", "year"]
        ).sort_index()
        single_y = single_panel["tea"].astype(float)
        single_x = single_panel[[var]].astype(float)
        single_x["constant"] = 1.0
        single_countries = single_data["country"].nunique()

        single_pooled = PooledOLS(single_y, single_x).fit(
            cov_type="clustered",
            cluster_entity=True,
        )
        single_random = RandomEffects(single_y, single_x).fit(
            cov_type="clustered",
            cluster_entity=True,
        )

        univariate_ols_columns[f"({number}) {var}"] = make_column(
            single_pooled,
            included_vars=[var],
            all_vars=displayed_vars,
            countries=single_countries,
        )
        univariate_re_columns[f"({number}) {var}"] = make_column(
            single_random,
            included_vars=[var],
            all_vars=displayed_vars,
            countries=single_countries,
        )

    univariate_ols_columns[f"({len(variables) + 1}) Joint"] = make_column(
        pooled,
        included_vars=displayed_vars,
        all_vars=displayed_vars,
        countries=countries,
    )
    univariate_re_columns[f"({len(variables) + 1}) Joint"] = make_column(
        random,
        included_vars=displayed_vars,
        all_vars=displayed_vars,
        countries=countries,
    )

    univariate_ols_table = pd.DataFrame(univariate_ols_columns)
    univariate_re_table = pd.DataFrame(univariate_re_columns)

    print(f"\nPOOLED OLS: UNIVARIATE AND JOINT (No country or year effects) — {sheet_name}")
    print(univariate_ols_table.round(4).to_string())

    print(f"\nRANDOM EFFECTS: UNIVARIATE AND JOINT (Country random effects; no year effects) — {sheet_name}")
    print(univariate_re_table.round(4).to_string())
    # print(f"\nRANDOM EFFECTS THETA — {sheet_name}")
    # print(random.theta.round(4).to_string())

    print(f"\nFIXED EFFECTS: UNIVARIATE AND JOINT (Country and year fixed effects) — {sheet_name}")
    print(fe_table.round(4).to_string())


# ==============================================================
# SECTION 1: RESULTS WITHOUT INTERPOLATION
# ==============================================================

print("\n" + "=" * 75)
print("RESULTS WITHOUT INTERPOLATION")
print("=" * 75)

for period, variables in VARIABLES.items():
    print(f"\nPERIOD — {period}")
    run_sample(f"S{period}", variables)


# ==============================================================
# SECTION 2: RESULTS WITH INTERPOLATION
# ==============================================================

print("\n" + "=" * 75)
print("RESULTS WITH INTERPOLATION")
print("=" * 75)

for period, variables in VARIABLES.items():
    print(f"\nPERIOD — {period}")
    run_sample(f"S{period} (I)", variables)

