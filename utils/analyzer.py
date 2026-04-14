def get_top_categories(df, column):
    result = df[column].value_counts().reset_index()
    result.columns = [column, "count"]
    return result.head(10)


def generate_basic_insight(result, column):
    if result.empty:
        return "No data available for insight."

    top_value = result.iloc[0][column]
    top_count = result.iloc[0]["count"]

    return f"The most common value in '{column}' is '{top_value}' with {top_count} records."