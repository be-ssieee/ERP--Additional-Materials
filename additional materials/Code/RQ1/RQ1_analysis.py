from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr, ttest_ind
from sklearn.metrics import cohen_kappa_score


# Files
score_file = "RQ1_dataset_clean.xlsx"
metadata_file = "RQ1_dataset_raw.xlsx"

output_dir = Path("outputs/rq1")
health_shelf_date = pd.Timestamp("2021-07-19", tz="UTC")

items = [
    "item1_clear_aims",
    "item2_reliable_sources",
    "item3_balanced_unbiased",
    "item4_additional_sources",
    "item5_uncertainty",
]

human_items = [
    "human_item1_clear_aims",
    "human_item2_reliable_sources",
    "human_item3_balanced_unbiased",
    "human_item4_additional_sources",
    "human_item5_uncertainty",
]


def group_summary(data, score_col):
    rows = []

    for status, label in [(1, "Health Source"), (0, "non-Health Source")]:
        x = data.loc[data["health_source"] == status, score_col]

        rows.append({
            "group": label,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(ddof=1),
            "median": x.median(),
        })

    return pd.DataFrame(rows)


def score_tests(data, score_col):
    health_source = data.loc[data["health_source"] == 1, score_col]
    non_health_source = data.loc[data["health_source"] == 0, score_col]

    mw = mannwhitneyu(
        health_source,
        non_health_source,
        alternative="two-sided"
    )

    welch = ttest_ind(
        health_source,
        non_health_source,
        equal_var=False
    )

    return {
        "mann_whitney_u": mw.statistic,
        "mann_whitney_p": mw.pvalue,
        "welch_t": welch.statistic,
        "welch_df": welch.df,
        "welch_p": welch.pvalue,
    }


def item_comparison(data, item_cols, run_fisher=True):
    rows = []

    for item in item_cols:
        health_source = data[data["health_source"] == 1]
        non_health_source = data[data["health_source"] == 0]

        row = {
            "item": item,
            "health_source_pass": int(health_source[item].sum()),
            "health_source_n": len(health_source),
            "health_source_pct": health_source[item].mean() * 100,
            "non_health_source_pass": int(non_health_source[item].sum()),
            "non_health_source_n": len(non_health_source),
            "non_health_source_pct": non_health_source[item].mean() * 100,
        }

        if run_fisher:
            table = pd.crosstab(data["health_source"], data[item])
            table = table.reindex(
                index=[1, 0],
                columns=[0, 1],
                fill_value=0
            )
            _, p = fisher_exact(table.values, alternative="two-sided")
            row["fisher_p"] = p

        rows.append(row)

    return pd.DataFrame(rows)


# Load final scoring data and video metadata
llm = pd.read_excel(score_file, sheet_name="LLM_scoring")
manual = pd.read_excel(score_file, sheet_name="Manual_scoring")
metadata = pd.read_excel(metadata_file)

metadata["published_at"] = pd.to_datetime(
    metadata["published_at"],
    errors="coerce",
    utc=True
)

# Keep only metadata for videos in the final 39-video scoring dataset
metadata_final = metadata[
    metadata["video_id"].isin(llm["video_id"])
][["video_id", "health_source", "published_at"]].drop_duplicates("video_id")

df = llm.merge(
    metadata_final,
    on="video_id",
    how="left",
    validate="one_to_one"
)

output_dir.mkdir(parents=True, exist_ok=True)


# Sample composition
df["post_health_shelf"] = df["published_at"] >= health_shelf_date

sample_composition = pd.DataFrame({
    "measure": [
        "Videos",
        "Published on/after 19 July 2021",
        "Published before 19 July 2021",
    ],
    "overall": [
        len(df),
        int(df["post_health_shelf"].sum()),
        int((~df["post_health_shelf"]).sum()),
    ],
    "health_source": [
        int((df["health_source"] == 1).sum()),
        int(((df["health_source"] == 1) & df["post_health_shelf"]).sum()),
        int(((df["health_source"] == 1) & ~df["post_health_shelf"]).sum()),
    ],
    "non_health_source": [
        int((df["health_source"] == 0).sum()),
        int(((df["health_source"] == 0) & df["post_health_shelf"]).sum()),
        int(((df["health_source"] == 0) & ~df["post_health_shelf"]).sum()),
    ],
})

print("\nRQ1 sample composition")
print(sample_composition.to_string(index=False))


# Human-LLM agreement
validation = manual.merge(
    llm[["video_id"] + items + ["mdiscern_total"]],
    on="video_id",
    how="left",
    validate="one_to_one"
)

agreement_rows = []

for human_item, llm_item in zip(human_items, items):
    raw_agreement = (
        validation[human_item] == validation[llm_item]
    ).mean()

    kappa = cohen_kappa_score(
        validation[human_item],
        validation[llm_item]
    )

    agreement_rows.append({
        "item": llm_item,
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "human_positive": int(validation[human_item].sum()),
        "llm_positive": int(validation[llm_item].sum()),
        "n": len(validation),
    })

agreement_table = pd.DataFrame(agreement_rows)

score_difference = (
    validation["human_total"] - validation["mdiscern_total"]
)

pearson = pearsonr(
    validation["human_total"],
    validation["mdiscern_total"]
)

spearman = spearmanr(
    validation["human_total"],
    validation["mdiscern_total"]
)

validation_summary = pd.DataFrame([{
    "n": len(validation),
    "exact_total_matches": int((score_difference == 0).sum()),
    "exact_total_match_pct": (score_difference == 0).mean() * 100,
    "within_1_point": int((score_difference.abs() <= 1).sum()),
    "within_1_point_pct": (score_difference.abs() <= 1).mean() * 100,
    "mae": score_difference.abs().mean(),
    "human_mean": validation["human_total"].mean(),
    "llm_mean": validation["mdiscern_total"].mean(),
    "pearson_r": pearson.statistic,
    "pearson_p": pearson.pvalue,
    "spearman_rho": spearman.statistic,
    "spearman_p": spearman.pvalue,
}])

print("\nHuman-LLM item agreement")
print(agreement_table.to_string(index=False))

print("\nHuman-LLM total-score validation")
print(validation_summary.to_string(index=False))


# Main Health Source vs non-Health Source analysis
main_descriptives = group_summary(df, "mdiscern_total")
main_tests = score_tests(df, "mdiscern_total")
main_items = item_comparison(df, items, run_fisher=True)

print("\nMain RQ1 descriptives")
print(main_descriptives.to_string(index=False))

print("\nMain score tests")
print(main_tests)

print("\nMain item-level comparisons")
print(main_items.to_string(index=False))


# Sensitivity analysis 1: independently human-coded videos
human_df = validation.merge(
    metadata_final,
    on="video_id",
    how="left",
    validate="one_to_one"
)

human_descriptives = group_summary(human_df, "human_total")
human_tests = score_tests(human_df, "human_total")
human_items_table = item_comparison(
    human_df,
    human_items,
    run_fisher=False
)

print("\nHuman-coded sensitivity analysis")
print(human_descriptives.to_string(index=False))
print(human_tests)

print("\nHuman-coded item pass rates")
print(human_items_table.to_string(index=False))


# Sensitivity analysis 2: videos published on/after 19 July 2021
post_df = df[
    df["published_at"] >= health_shelf_date
].copy()

post_descriptives = group_summary(
    post_df,
    "mdiscern_total"
)

post_tests = score_tests(
    post_df,
    "mdiscern_total"
)

post_items = item_comparison(
    post_df,
    items,
    run_fisher=True
)

print("\nPost-Health-Shelf sensitivity analysis")
print(post_descriptives.to_string(index=False))
print(post_tests)

print("\nPost-Health-Shelf item-level comparisons")
print(post_items.to_string(index=False))


# Save analysis tables
sample_composition.to_csv(
    output_dir / "rq1_sample_composition.csv",
    index=False
)

agreement_table.to_csv(
    output_dir / "rq1_human_llm_item_agreement.csv",
    index=False
)

validation_summary.to_csv(
    output_dir / "rq1_human_llm_validation_summary.csv",
    index=False
)

main_descriptives.to_csv(
    output_dir / "rq1_main_descriptives.csv",
    index=False
)

main_items.to_csv(
    output_dir / "rq1_main_item_comparisons.csv",
    index=False
)

pd.DataFrame([main_tests]).to_csv(
    output_dir / "rq1_main_score_tests.csv",
    index=False
)

human_descriptives.to_csv(
    output_dir / "rq1_human_only_descriptives.csv",
    index=False
)

human_items_table.to_csv(
    output_dir / "rq1_human_only_item_pass_rates.csv",
    index=False
)

pd.DataFrame([human_tests]).to_csv(
    output_dir / "rq1_human_only_score_tests.csv",
    index=False
)

post_descriptives.to_csv(
    output_dir / "rq1_post_health_shelf_descriptives.csv",
    index=False
)

post_items.to_csv(
    output_dir / "rq1_post_health_shelf_item_comparisons.csv",
    index=False
)

pd.DataFrame([post_tests]).to_csv(
    output_dir / "rq1_post_health_shelf_score_tests.csv",
    index=False
)

print(f"\nSaved RQ1 outputs to: {output_dir}")
