import os


class Path:
    current_path = os.getcwd()

    # Directories
    data_path = os.path.join(current_path, "data")
    datasets = os.path.join(data_path, "datasets")

    # Models folder
    models = os.path.join(data_path, "models")
    models_filename = os.path.join(models, "model.keras")
    models_json = os.path.join(models, "classes.json")

    # Image for predict
    predict_path = os.path.join(data_path, "predict")

    # Graphs folder
    graphs = os.path.join(data_path, "graphs")
    graphs_images_count = os.path.join(graphs, "images_count.png")
    graphs_images_balances_count = os.path.join(graphs, "images_count_balances.png")
    graphs_model_train_history = os.path.join(graphs, "model_train_history.png")
    classification_report_path = os.path.join(graphs, "classification_report_path.txt")

    # Mushrooms image main folder
    mushrooms = os.path.join(datasets, "Mushrooms")

    # Create neccesery folders
    for directory in [data_path, models, datasets, graphs, predict_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)