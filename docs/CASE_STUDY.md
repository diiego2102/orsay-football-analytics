# Football forecasting case study

## Decision
Convert historical match outcomes into explicit pre-match probabilities and assess whether a more elaborate model improves on a simple reference.

## Implementation
Recovered MEC-1.2.1 combines past-only rolling form, pre-match Elo, four Poisson regressors and a Dixon–Coles score matrix. A new offline harness checks real historical data, isolates complete seasons and compares predictions with a training-average Poisson model. Original source versions remain unchanged.

## Demonstrated result
The 380-match holdout improves log loss and outcome accuracy, but worsens goal-count MAE. See the full benchmark and CSV rather than treating model complexity as evidence of quality.

## Project evolution
Orsay began as an independent football analytics project using Solver, Poisson and Monte Carlo, with free and paid prediction content. The recovered Python MEC implementation and current public benchmark are later technical evolution. RBDT-26 remains an experimental mathematical concept; no runnable recovered implementation or measured RBDT performance is claimed.

## Next decision
Test multiple rolling seasons, remove numeric identifier features, inspect promoted-team handling and calibrate probabilities on training-only validation before using any held-out comparison for model selection.
