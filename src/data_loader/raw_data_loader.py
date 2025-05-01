import pandas as pd

from src.utils.decorators import start_finish_function


@start_finish_function
def get_df(engine, columns_datatime: list = ['date_create', 'date_fakticheskaya_vygruzki']) -> pd.DataFrame:
    query = """
        select *
        from model_upp_work.without_days_of_delivery
    """
    return pd.read_sql(query, engine, parse_dates=columns_datatime)
