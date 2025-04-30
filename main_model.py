from data_loader.raw_data_loader import get_df
from preprocessor.preprocessing import ManagerPreprocessing
from setting.settings import setup_logger, dbm
from utils.decorators import start_finish_function


logger = setup_logger()


@start_finish_function
def start_program():
    logger.info(f"Старт программы")

    engine = dbm.get_engine()

    df = get_df(engine)
    feature_name = ['delivery_point', 'rasstoyanie', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki', 'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'date_create', 'tonnazh', 'obem_znt', 'kolvo_gruzovykh_mest', 'lt_stoimost_perevozki']

    df = ManagerPreprocessing(df=df).df_preprocess(columns_dropna=feature_name, set_correct_datetype=True)
    return df

    # X_train, y_train, X_test, y_test = Preprocessing.df_split_test_and_train(df)


if __name__ == '__main__':
    start_program()