args <- commandArgs(trailingOnly = TRUE)

data_path <- if (length(args) >= 1) {
  args[1]
} else {
  "data/RQ2_comments_predictions.xlsx"
}

output_dir <- if (length(args) >= 2) {
  args[2]
} else {
  "outputs/rq2_pretrend"
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

library(readxl)
library(dplyr)
library(lubridate)
library(clubSandwich)

comments <- read_excel(data_path)

required_columns <- c(
  "group",
  "channel",
  "comment_published_at",
  "period",
  "BERT_MCD_prob",
  "BERT_MCD_pred",
  "BERT_agreement_prob",
  "BERT_agreement_pred"
)

missing_columns <- setdiff(required_columns, names(comments))

if (length(missing_columns) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

valid_groups <- c("treatment", "control")
observed_groups <- unique(tolower(as.character(comments$group)))
unexpected_groups <- setdiff(observed_groups, valid_groups)

if (length(unexpected_groups) > 0) {
  stop(
    paste(
      "Unexpected group values:",
      paste(unexpected_groups, collapse = ", ")
    )
  )
}

comments <- comments %>%
  mutate(
    group = tolower(as.character(group)),
    period = toupper(as.character(period)),
    Treatment = if_else(group == "treatment", 1, 0),
    comment_published_at = parse_date_time(
      as.character(comment_published_at),
      orders = c(
        "Y-m-d H:M:S",
        "Y-m-dTH:M:Sz",
        "Y-m-dTH:M:SOS"
      ),
      tz = "UTC",
      quiet = TRUE
    )
  )

if (any(is.na(comments$comment_published_at))) {
  stop("Some comment_published_at values could not be parsed.")
}

pre <- comments %>%
  filter(
    comment_published_at >= ymd_hms(
      "2019-07-19 00:00:00",
      tz = "UTC"
    ),
    comment_published_at < ymd_hms(
      "2021-07-19 00:00:00",
      tz = "UTC"
    )
  ) %>%
  mutate(
    month = floor_date(comment_published_at, unit = "month")
  )

channel_month <- pre %>%
  group_by(channel, group, Treatment, month) %>%
  summarise(
    MCD_share = mean(BERT_MCD_pred),
    MCD_score = mean(BERT_MCD_prob),
    Agreement_share = mean(BERT_agreement_pred),
    Agreement_score = mean(BERT_agreement_prob),
    n_comments = n(),
    .groups = "drop"
  ) %>%
  arrange(month, channel) %>%
  mutate(
    month_factor = factor(format(month, "%Y-%m")),
    channel_factor = factor(channel)
  )

first_month <- min(channel_month$month)

channel_month <- channel_month %>%
  mutate(
    trend = (
      year(month) * 12 + month(month)
      - (year(first_month) * 12 + month(first_month))
    )
  )

write.csv(
  channel_month,
  file.path(output_dir, "rq2_channel_month_pretrend.csv"),
  row.names = FALSE
)

cat(
  "Channel-month observations:",
  nrow(channel_month),
  "\n"
)

if (nrow(channel_month) != 213) {
  warning(
    paste(
      "Expected 213 channel-month observations but found",
      nrow(channel_month)
    )
  )
}

run_pretrend <- function(data, outcome) {

  formula_text <- paste0(
    outcome,
    " ~ channel_factor + month_factor + Treatment:trend"
  )

  model <- lm(
    as.formula(formula_text),
    data = data
  )

  V_CR2 <- vcovCR(
    model,
    cluster = data$channel,
    type = "CR2"
  )

  test <- as.data.frame(
    coef_test(
      model,
      vcov = V_CR2,
      test = "Satterthwaite"
    )
  )

  interval <- as.data.frame(
    conf_int(
      model,
      vcov = V_CR2,
      test = "Satterthwaite"
    )
  )

  target <- test[
    test$Coef == "Treatment:trend",
    ,
    drop = FALSE
  ]

  target_ci <- interval[
    interval$Coef == "Treatment:trend",
    ,
    drop = FALSE
  ]

  if (nrow(target) != 1 || nrow(target_ci) != 1) {
    stop(
      paste(
        "Could not uniquely identify Treatment:trend for",
        outcome
      )
    )
  }

  data.frame(
    outcome = outcome,
    beta = target$beta,
    SE = target$SE,
    df = target$df_Satt,
    p_value = target$p_Satt,
    CI_low = target_ci$CI_L,
    CI_high = target_ci$CI_U,
    row.names = NULL
  )
}

outcomes <- c(
  "MCD_share",
  "MCD_score",
  "Agreement_share",
  "Agreement_score"
)

results <- do.call(
  rbind,
  lapply(
    outcomes,
    function(x) run_pretrend(channel_month, x)
  )
)

write.csv(
  results,
  file.path(output_dir, "rq2_pretrend_results.csv"),
  row.names = FALSE
)

cat("\nPre-intervention differential trend results\n")
print(results, row.names = FALSE)
