from typing import Tuple, Optional
import pandas as pd
from datetime import timedelta

from src.utils.decorators import start_finish_function


# Надо сюда закинуть все сплитеры


class FixedDateTrainTestSplitter:
    def __init__(self, test_days: int = 90, date_column: str = 'date_create'):
        self.test_days = test_days
        self.date_column = date_column

    @start_finish_function
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy().sort_values(self.date_column)
        split_date = df[self.date_column].max() - timedelta(self.test_days)

        train_df = df[df[self.date_column] < split_date]
        test_df = df[df[self.date_column] >= split_date]
        return train_df, test_df


class TimeBasedSubsetSelector:
    def __init__(self, percent: float = 0.4, n_segments: int = 5, date_column: str = 'date_create'):
        self.percent = percent
        self.n_segments = n_segments
        self.date_column = date_column

    @start_finish_function
    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(by=self.date_column)
        total_len = len(df)
        segment_size = int((total_len * self.percent) / self.n_segments)
        step = (total_len - segment_size) // (self.n_segments - 1)

        indices = []
        for i in range(self.n_segments):
            start_idx = i * step
            end_idx = start_idx + segment_size
            indices.extend(range(start_idx, end_idx))

        return df.iloc[indices]


class FeatureTargetSplitter:
    def __init__(self, target_column: str = 'y'):
        self.target_column = target_column

    @start_finish_function
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return df.drop([self.target_column], axis=1), df[self.target_column]


class SplitManager:
    def __init__(self,
                 date_splitter: FixedDateTrainTestSplitter,
                 subset_selector: Optional[TimeBasedSubsetSelector] = None,
                 target_column: str = 'y'):
        self.date_splitter = date_splitter
        self.subset_selector = subset_selector
        self.feature_target_splitter = FeatureTargetSplitter(target_column=target_column)

    @start_finish_function
    def run(self, df: pd.DataFrame, use_subset: bool = False) -> Tuple:
        df_train, df_test = self.date_splitter.split(df)

        if use_subset and self.subset_selector:
            df_train = self.subset_selector.select(df_train)

        x_train, y_train = self.feature_target_splitter.split(df_train)
        x_test, y_test = self.feature_target_splitter.split(df_test)

        return x_train, y_train, x_test, y_test

