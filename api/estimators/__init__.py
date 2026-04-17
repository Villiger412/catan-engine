from .base import EstimateInput, EstimateOutput, WinEstimator
from .monte_carlo import MonteCarloEstimator

# Registry — add new estimators here; the API and UI pick them up automatically.
ESTIMATORS: dict[str, WinEstimator] = {
    "monte_carlo": MonteCarloEstimator(),
}
