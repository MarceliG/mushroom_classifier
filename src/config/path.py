import os


class Path:
    current_path = os.getcwd()

    # Directories
    data_path = os.path.join(current_path, "data")
    datasets = os.path.join(data_path, "datasets")

    # Models folder
    models = os.path.join(data_path, "models")
    models_filename = os.path.join(models, "model.keras")

    # Graphs folder
    graphs = os.path.join(data_path, "graphs")
    graphs_images_count = os.path.join(graphs, "images_count.png")
    graphs_images_balances_count = os.path.join(graphs, "images_count_balances.png")
    graphs_model_train_history = os.path.join(graphs, "model_train_history.png")

    # Mushrooms image main folder
    mushrooms = os.path.join(datasets, "Mushrooms")

    # Create neccesery folders
    for directory in [data_path, models, datasets, graphs]:
        if not os.path.exists(directory):
            os.makedirs(directory)
