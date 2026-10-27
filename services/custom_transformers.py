import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(
    BaseEstimator,
    TransformerMixin
):
    def fit(
        self,
        X,
        y=None
    ):
        X_df = pd.DataFrame(X).copy()

        self.columns_ = list(
            X_df.columns
        )

        self.maps_ = {}

        for col in self.columns_:

            values = (
                X_df[col]
                .astype("string")
                .fillna(
                    "__MISSING__"
                )
            )

            self.maps_[col] = (
                values
                .value_counts(
                    normalize=True
                )
                .to_dict()
            )

        return self

    def transform(
        self,
        X
    ):
        X_df = pd.DataFrame(
            X,
            columns=self.columns_
        )

        out = pd.DataFrame(
            index=X_df.index
        )

        for col in self.columns_:

            values = (
                X_df[col]
                .astype("string")
                .fillna(
                    "__MISSING__"
                )
            )

            out[col] = (
                values
                .map(
                    self.maps_[col]
                )
                .fillna(0.0)
                .astype(float)
            )

        return out.to_numpy()

    def get_feature_names_out(
        self,
        input_features=None
    ):
        if input_features is None:
            input_features = self.columns_

        return np.asarray(
            [
                f"{c}__freq"
                for c in input_features
            ],
            dtype=object
        )