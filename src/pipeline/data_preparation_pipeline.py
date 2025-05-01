from typing import List, Tuple

import pandas as pd

from src.preprocessor.preprocessing import ManagerPreprocessing
from src.splitter.splitter import SplitManager, FixedDateTrainTestSplitter, TimeBasedSubsetSelector
from src.utils.decorators import start_finish_function


class TrainTestPreparer:
    def __init__(self,
                 feature_columns: List[str],
                 test_days: int = 90,
                 subset_percent: float = 0.4,
                 n_segments: int = 5,
                 target_column: str = 'y',
                 date_column: str = 'date_create'):
        self.feature_columns = feature_columns
        self.preprocessor = ManagerPreprocessing
        self.split_manager = SplitManager(
            date_splitter=FixedDateTrainTestSplitter(test_days=test_days, date_column=date_column),
            subset_selector=TimeBasedSubsetSelector(percent=subset_percent, n_segments=n_segments,
                                                    date_column=date_column),
            target_column=target_column
        )

    @start_finish_function
    def prepare(self, df: pd.DataFrame) -> Tuple:
        df = self.preprocessor(df).df_preprocess(columns_dropna=self.feature_columns, set_correct_datetype=True)

        x_train, y_train, x_test, y_test = self.split_manager.run(df, use_subset=False)
        x_subset, y_subset, _, _ = self.split_manager.run(df, use_subset=True)

        return x_train, y_train, x_test, y_test, x_subset, y_subset
