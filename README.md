# ERP Additional Materials

## 1. Project Overview

This repository contains the code and supporting materials for the MSc Data Science Extended Research Project *Health Communication and Misinformation: Evaluating YouTube’s Health Source Initiative through Video Information Quality and Audience Discussion*.

## 2. Data Requirements

The full YouTube datasets are not included in this public repository. To reproduce the reported analyses, the archived RQ1 scoring/metadata datasets and RQ2 labelled/comment-prediction datasets are required. 
Because YouTube content and API results may change over time, re-collecting the data may not reproduce the exact historical dataset used in the dissertation.
The original data collection and preparation procedures are described in the Technical Appendix. YouTube data were collected using the YouTube Data API, with video transcripts retrieved using the `youtube-transcript-api` Python package. Please contact 2582425194@qq.com if you need the archived dataset.

## 3. Repository File Guide

- `Code/RQ1_analysis.py` — RQ1 validation, main tests and sensitivity analyses.
- `Code/RQ2_classifier.py` — BERT training, validation-threshold selection, test evaluation and full-data prediction.
- `Code/RQ2_did_analysis.py` — RQ2 descriptive shares and DiD estimation.
- `Code/RQ2_pretrend.R` — channel-month pre-intervention trend analysis.
- `Documentation/RQ1_mDISCERN coding rules.docx` — five-item mDISCERN coding rules.
- `Documentation/RQ1_LLM-assisted coding prompt.docx` — LLM-assisted coding prompt and recorded configuration.
- `Supporting materials-outputs/` — principal reported results for verification.
- `requirements.txt` — required Python packages.

## 4. Environment Setup

Install the required Python packages from the repository root:

`pip install -r requirements.txt`

The pre-intervention trend analysis additionally requires the R packages `readxl`, `dplyr`, `lubridate`, and `clubSandwich`.

## 5. Running the Analysis

Run the analyses from the repository root in the following order:

1. `python code/RQ1_analysis.py`
2. `python code/RQ2_classifier.py`
3. `python code/RQ2_did_analysis.py`
4. `Rscript code/RQ2_pretrend.R`

Detailed input requirements and parameter settings are documented in the Technical Appendix and within the scripts.
