from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from typing import Optional, List, Dict, Tuple

from setting.settings import setup_logger
from utils.decorators import start_finish_function
from abc import ABC, abstractmethod

logger = setup_logger()


class DatetimePreprocessor:
    def __init__(self, datetime_columns: List[str], col_decompose: str):
        self.datetime_columns = datetime_columns
        self.col_decompose = col_decompose

    @start_finish_function
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.datetime_columns:
            df[col] = pd.to_datetime(df[col])

            if self.col_decompose == col:
                df['year'] = df[col].dt.year
                df['month'] = df[col].dt.month

        return df.sort_values(self.col_decompose)


class TargetCreator:
    @start_finish_function
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_target = df.copy()
        df_target['days_by_distance'] = np.ceil(df_target['rasstoyanie'] / 450).astype(int)

        # 2. Общее количество рабочих дней доставки
        df_target['working_days_total'] = df_target['days_by_distance'] + df_target['delivery_point'] * 5

        # 3. Плановая дата доставки: прибавляем рабочие дни (пропуская выходные)
        df_target['planned_delivery_date'] = df_target.apply(
            lambda row: row['date_create'] + BDay(row['working_days_total']),
            axis=1
        )

        df['planned_delivery_days'] = (df_target['planned_delivery_date'] - df_target['date_create']).dt.days.astype(int)
        df['y'] = (df_target['planned_delivery_date'] < df_target['date_fakticheskaya_vygruzki']).astype(int)

        return df


class RareCityDropper:
    def __init__(self, city_columns: List[str], min_count: int = 5):
        self.city_columns = city_columns
        self.min_count = min_count

    @start_finish_function
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.city_columns:
            counts = df[col].value_counts()
            valid_values = counts[counts > self.min_count].index
            df = df[df[col].isin(valid_values)]
        return df


class CategoryFeatureEngineer:
    @staticmethod
    def assign_group(distance):
        if pd.isna(distance):
            return np.nan
        if distance == 0:
            return 0
        elif distance < 1000:
            return (distance // 100) * 100
        else:
            return (distance // 1000) * 1000

    @staticmethod
    def tonnazh_group(tonnazh):
        if pd.isna(tonnazh):
            return np.nan
        if tonnazh == 0:
            return 0
        elif tonnazh < 1:
            return round((tonnazh // 0.1) * 0.1, 1)
        elif tonnazh < 5:
            return (tonnazh // 1) * 1
        elif tonnazh < 10:
            return 5
        else:
            return 10

    @staticmethod
    def cost_group(cost):
        if pd.isna(cost):
            return np.nan
        if cost == 0:
            return 0
        elif cost < 10000:
            return (cost // 1000) * 1000
        elif cost < 50000:
            return (cost // 10000) * 10000
        elif 50000 < cost < 150000:
            return (cost // 50000) * 50000
        elif 150000 < cost < 500001:
            return 150000
        else:
            return 500000

    @start_finish_function
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['distance_group'] = df['rasstoyanie'].map(self.assign_group).astype(int)
        df['tonnazh_group'] = df['tonnazh'].map(self.tonnazh_group)
        df['cost_group'] = df['lt_stoimost_perevozki'].map(self.cost_group).astype(int)
        return df


class GeospatialFeatureEngineer:
    R = 6371.0

    @start_finish_function
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lat1 = np.radians(df['lat_zagruzki'].values)
        lon1 = np.radians(df['lng_zagruzki'].values)
        lat2 = np.radians(df['lat_vygruzki'].values)
        lon2 = np.radians(df['lng_vygruzki'].values)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        df['geo_rasstoyanie_km'] = self.R * c
        return df


class BasePreprocessing(ABC):
    def __init__(self, df: pd.DataFrame,
                 datetime_columns: Optional[List[str]] = None,
                 int_columns: Optional[List[str]] = None,
                 float_columns: Optional[List[str]] = None,
                 object_columns: Optional[List[str]] = None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df должен быть экземпляром pd.DataFrame")

        self._df = df.copy()

        self.datetime_columns = datetime_columns or self._df.select_dtypes(include=[np.datetime64]).columns.tolist()
        self.int_columns = int_columns or self._df.select_dtypes(include=[np.int64, int]).columns.tolist()
        self.float_columns = float_columns or self._df.select_dtypes(include=[np.float64, float]).columns.tolist()
        self.object_columns = object_columns or self._df.select_dtypes(include=['object']).columns.tolist()

        self.type_mapping = {
            'datetime': self.datetime_columns,
            'int': self.int_columns,
            'float': self.float_columns,
            'object': self.object_columns
        }

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        logger.error(f"Ошибка: метод preprocess() не реализован в {self.__class__.__name__}")

    def get_df(self) -> pd.DataFrame:
        return self._df.copy()

    def df_preprocess(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs).get_df()

    @start_finish_function
    def set_correct_datetype(self, type_mapping: Dict[str, List[str]]) -> None:

        for dtype, columns in type_mapping.items():
            if not columns:
                continue
            for col in columns:
                if dtype == 'datetime':
                    self._df[col] = pd.to_datetime(self._df[col])
                else:
                    self._df[col] = self._df[col].astype(dtype)

    @start_finish_function
    def _dropna(self, col_names: list):
        count_col = self._df.shape[0]
        self._df = self._df.dropna(subset=col_names)
        logger.info(f'Удалено пустых столбцов: {count_col - self._df.shape[0]} шт, осталось {self._df.shape[0]}')


class ManagerPreprocessing(BasePreprocessing):
    def __init__(self, df: pd.DataFrame,
                 datetime_columns: Optional[List[str]] = None,
                 int_columns: Optional[List[str]] = None,
                 float_columns: Optional[List[str]] = None,
                 object_columns: Optional[List[str]] = None):
        super().__init__(df, datetime_columns, int_columns, float_columns, object_columns)

    @start_finish_function
    def preprocess(self, columns_dropna: List[str] = None, set_correct_datetype: bool = False) -> 'ManagerPreprocessing':
        if isinstance(columns_dropna, list):
            self._dropna(columns_dropna)

        self._df = DatetimePreprocessor(self.datetime_columns, 'date_create').transform(self._df)
        self._df = TargetCreator().transform(self._df)
        self._df = RareCityDropper(['region_zagruzki', 'region_vygruzki']).transform(self._df)
        self._df = CategoryFeatureEngineer().transform(self._df)
        self._df = GeospatialFeatureEngineer().transform(self._df)

        if set_correct_datetype:
            self.set_correct_datetype(self.type_mapping)
        return self
