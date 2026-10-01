# Development roadmap

## Orsay

1. Publish this recovered baseline after the code and presentation review.
2. Reproduce historical evaluation with a documented dataset and chronological holdout.
3. Compare against league-average goal and simple Elo baselines; report log loss, Brier score, goal MAE and calibration.
4. Refactor the recovered engine into ingestion, features, model, evaluation and reporting modules without changing predictions unintentionally.
5. Build a compact results dashboard with probability distributions and uncertainty.
6. Recover any RBDT-26 source separately and evaluate its role before integrating it.
7. Add richer data only when the source, access rights and contribution to model quality are demonstrated.

## Complementary portfolio projects

### Operations Analytics Dashboard

Build a new generic implementation with synthetic operational records. Demonstrate Python/SQL transformations, documented KPIs, anomaly flags, filtering and an executive view. Use operational experience to guide the problem; do not copy employer databases, infrastructure, internal business rules or restricted code. Include a data dictionary, screenshots and instructions to run locally.

### Billing Automation & Data Quality

Build a generic synthetic workflow for periodic readings, billing-cycle validation and exception handling. Demonstrate completeness checks, duplicate detection, abnormal-consumption flags and a review report. Describe the historical billing time improvement separately as a professional achievement; do not attribute that improvement to this new demonstration.

## Existing learning material

Pacífico material includes a consumption-forecasting notebook using Random Forest and a cost-minimisation notebook using PuLP. Review project authorship, input data and evaluation before adapting either. The forecasting notebook includes code that labels precision as specificity; that analytical mistake should be corrected in any derivative, not copied into a public report.

The provided professional archives contain extensive dashboards and automation material, including multi-gigabyte ZIP/7z files. Retain originals privately. Catalogue source files and extract only the necessary components before preparing a new generic implementation.
