def create_features(df):
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    df['is_weekend'] = (df['arrival_date_day_of_month'] % 7 == 0).astype(int)
    return df
