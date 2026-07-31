import lightgbm as lgb
import numpy as np


class OneShotAllocator:
    """Predict per-rung quality gap, then choose by argmin predicted loss.

    Regression rather than classification: a near-tie mistake between two
    close rungs costs almost nothing, while a genuine misclassification costs
    the full difference. See DESIGN.md §9.

    Uses LightGBM's native train()/Dataset API, not the sklearn wrapper — no
    scikit-learn dependency (IMPLEMENTATION.md §1).
    """

    def __init__(
        self, rung_names: list[str], lgbm_params: dict | None = None, num_boost_round: int = 100
    ):
        self.rung_names = rung_names
        self.lgbm_params = {"verbosity": -1, "min_data_in_leaf": 20, **(lgbm_params or {})}
        self.num_boost_round = num_boost_round
        self.models: dict[str, lgb.Booster] = {}
        self.mean_time: dict[str, float] = {}

    def fit(
        self,
        X: np.ndarray,
        gaps_per_rung: dict[str, np.ndarray],
        times_per_rung: dict[str, np.ndarray],
    ) -> None:
        for rung in self.rung_names:
            dataset = lgb.Dataset(X, label=gaps_per_rung[rung])
            self.models[rung] = lgb.train(
                self.lgbm_params, dataset, num_boost_round=self.num_boost_round
            )
            self.mean_time[rung] = float(np.mean(times_per_rung[rung]))

    def choose(self, x: np.ndarray, lambda_: float) -> str:
        x_row = x.reshape(1, -1)
        predicted_loss = {
            rung: self.models[rung].predict(x_row)[0] + lambda_ * self.mean_time[rung]
            for rung in self.rung_names
        }
        return min(predicted_loss, key=predicted_loss.get)
