def clean_data(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna()
    return df