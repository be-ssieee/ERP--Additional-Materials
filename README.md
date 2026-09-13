# ERP--Additional-Materials
1. Project Overview
This repository contains the code and supporting materials for the MSc Data Science Extended Research Project Health Communication and Misinformation: Evaluating YouTube’s Health Source Initiative through Video Information Quality and Audience Discussion.

2.Data Requirements
The full YouTube datasets are not included in this public repository. To reproduce the reported analyses, the archived RQ1 scoring/metadata datasets and RQ2 labelled/comment-prediction datasets are required. The original data collection and preparation procedures are described in the Technical Appendix. YouTube data were collected using the YouTube Data API, with video transcripts retrieved using the youtube-transcript-api Python package.

3.Repository File Guide
code/rq1_analysis.py - RQ1 validation, main tests and sensitivity analyses.
code/rq2_classifier.py - BERT training, validation-threshold selection, test evaluation and full-data prediction.
code/rq2_did_analysis.py - RQ2 descriptive shares and DiD estimation.
code/rq2_pretrend.R - channel-month pre-intervention trend analysis.
documentation/rq1_mdiscern coding rules.docx - five-item mDISCERN coding rules.
documentation/rq1_llm-assisted_prompt.docx - LLM-assisted coding prompt and recorded configuration.
Supporting materials-outputs
requirements.txt - required Python packages.
README.md - repository overview, data instructions and run order.

4.Running the analysis
Run the analyses from the repository root in the following order:
python code/RQ1_analysis.py
python code/RQ2_classifier.py
python code/RQ2_did_analysis.py
Rscript code/RQ2_pretrend.R
Detailed input requirements and parameter settings are documented in the Technical Appendix and within the scripts.
