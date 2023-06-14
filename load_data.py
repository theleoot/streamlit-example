def load_data(file_name, columns_name):
    return pd.read_csv("/content/drive/MyDrive/Project/data.csv", names=columns_names, delimiter=r"\s+", header=None,
                       index_col=False)
