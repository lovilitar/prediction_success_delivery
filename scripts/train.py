from src.data_loader.raw_data_loader import get_df
from src.pipeline.data_preparation_pipeline import TrainTestPreparer
from src.setting.settings import setup_logger, dbm
from src.utils.decorators import start_finish_function


logger = setup_logger()


@start_finish_function
def start_program():
    logger.info(f"Старт программы")

    engine = dbm.get_engine()

    df = get_df(engine)
    feature_name = ['delivery_point', 'rasstoyanie', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki', 'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'date_create', 'tonnazh', 'obem_znt', 'kolvo_gruzovykh_mest', 'lt_stoimost_perevozki']

    x_train, y_train, x_test, y_test, x_subset, y_subset = TrainTestPreparer(feature_columns=feature_name).prepare(df)
    print(x_train.shape)

    return df


if __name__ == '__main__':
    start_program()
