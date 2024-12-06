import os

class Path:
    current_path = os.getcwd()
    # directories
    data_path = os.path.join(current_path, "data")
    datasets = os.path.join(data_path, "datasets")
    models = os.path.join(data_path, "models")
    mushrooms = os.path.join(datasets, "Mushrooms")

    # Create neccesery folders
    for directory in [
        data_path,
        models,
        datasets
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)