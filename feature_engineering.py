def create_features(df):
    if "arrival_date_day_of_month" in df.columns:
        df['is_weekend'] = (df['arrival_date'] % 7 == 0).astype(int)
    else:
        df['is_weekend']=0
    return df
