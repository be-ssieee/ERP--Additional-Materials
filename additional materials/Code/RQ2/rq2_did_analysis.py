
from pathlib import Path
import argparse

import pandas as pd
import statsmodels.formula.api as smf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="RQ2_comments_predictions.xlsx",
        help="Path to the final 52,541-comment prediction dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rq2_did",
        help="Directory for reproduced tables.",
    )
    return parser.parse_args()


def fit_did(data, outcome):
    model = smf.ols(
        f"{outcome} ~ Treatment + Post + Treatment:Post",
        data=data,
    ).fit()

    hc1 = model.get_robustcov_results(cov_type="HC1")

    clustered = model.get_robustcov_results(
        cov_type="cluster",
        groups=data["video_id"],
        use_correction=True,
    )

    term_names = model.model.exog_names
    interaction_index = term_names.index("Treatment:Post")

    clustered_ci = clustered.conf_int()[interaction_index]
    hc1_ci = hc1.conf_int()[interaction_index]

    return {
        "outcome": outcome,
        "beta_DiD": model.params["Treatment:Post"],
        "HC1_SE": hc1.bse[interaction_index],
        "HC1_CI_low": hc1_ci[0],
        "HC1_CI_high": hc1_ci[1],
        "HC1_p": hc1.pvalues[interaction_index],
        "cluster_SE": clustered.bse[interaction_index],
        "cluster_CI_low": clustered_ci[0],
        "cluster_CI_high": clustered_ci[1],
        "cluster_p": clustered.pvalues[interaction_index],
        "N": int(model.nobs),
        "video_clusters": int(data["video_id"].nunique()),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.data)

    required = {
        "group",
        "video_id",
        "period",
        "BERT_MCD_pred",
        "BERT_agreement_pred",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["group"] = data["group"].astype(str).str.lower()
    data["period"] = data["period"].astype(str).str.upper()

    data["Treatment"] = (data["group"] == "treatment").astype(int)
    data["Post"] = (data["period"] == "AFTER").astype(int)

    outcomes = {
        "MCD": "BERT_MCD_pred",
        "Agreement": "BERT_agreement_pred",
    }

    # Descriptive outcome shares by group and period.
    descriptive_rows = []

    for label, outcome in outcomes.items():
        shares = (
            data.groupby(["group", "period"], observed=True)[outcome]
            .agg(["mean", "count"])
            .reset_index()
        )

        for _, row in shares.iterrows():
            descriptive_rows.append(
                {
                    "outcome": label,
                    "group": row["group"],
                    "period": row["period"],
                    "share": row["mean"],
                    "percent": 100 * row["mean"],
                    "N_comments": int(row["count"]),
                }
            )

    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(
        output_dir / "rq2_descriptive_shares.csv",
        index=False,
    )

    # Raw Difference-in-Differences from the four cell means.
    did_rows = []

    for label, outcome in outcomes.items():
        cell = (
            data.groupby(["group", "period"], observed=True)[outcome]
            .mean()
        )

        treatment_before = cell.loc[("treatment", "BEFORE")]
        treatment_after = cell.loc[("treatment", "AFTER")]
        control_before = cell.loc[("control", "BEFORE")]
        control_after = cell.loc[("control", "AFTER")]

        treatment_change = treatment_after - treatment_before
        control_change = control_after - control_before
        did = treatment_change - control_change

        did_rows.append(
            {
                "outcome": label,
                "Treatment_Before": treatment_before,
                "Treatment_After": treatment_after,
                "Treatment_change": treatment_change,
                "Control_Before": control_before,
                "Control_After": control_after,
                "Control_change": control_change,
                "DiD": did,
                "DiD_percentage_points": 100 * did,
            }
        )

    raw_did = pd.DataFrame(did_rows)
    raw_did.to_csv(
        output_dir / "rq2_raw_did.csv",
        index=False,
    )

    # Regression-based DiD.
    regression_results = []

    for label, outcome in outcomes.items():
        result = fit_did(data, outcome)
        result["outcome"] = label
        regression_results.append(result)

    regression_table = pd.DataFrame(regression_results)
    regression_table.to_csv(
        output_dir / "rq2_did_regression.csv",
        index=False,
    )

    print("\nDescriptive outcome shares (%)")
    print(
        descriptive[
            ["outcome", "group", "period", "percent", "N_comments"]
        ].to_string(index=False)
    )

    print("\nRaw DiD (percentage points)")
    print(
        raw_did[
            ["outcome", "DiD_percentage_points"]
        ].to_string(index=False)
    )

    print("\nRegression DiD: video-clustered inference")
    print(
        regression_table[
            [
                "outcome",
                "beta_DiD",
                "cluster_SE",
                "cluster_CI_low",
                "cluster_CI_high",
                "cluster_p",
                "N",
                "video_clusters",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
