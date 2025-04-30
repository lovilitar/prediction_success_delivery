

# Надо сюда закинуть все сплитеры

class Preprocessing:
    @staticmethod
    def df_split_test_and_train(
        df, date_column='date_create',
        test_count_days=90
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        df_split = df.copy()
        df_split = df_split.sort_values(date_column)

        split_date = df_split[date_column].max() - timedelta(test_count_days)

        train_df = df_split[df_split[date_column] < split_date]
        test_df = df_split[df_split[date_column] >= split_date]

        X_train = train_df.drop(['y'], axis=1)
        y_train = train_df.y

        X_test = test_df.drop(['y'], axis=1)
        y_test = test_df.y

        return X_train, y_train, X_test, y_test