import os
import zipfile
import kaggle

class Manager:

    @staticmethod
    def download_dataset(author:str, dataset_name:str, save_loaction:str) -> None:
        # Need Kaggle account to dowload it
        api = kaggle.KaggleApi()
        api.authenticate()

        # Download
        full_dataset_name = f"{author}/{dataset_name}"
        api.dataset_download_files(full_dataset_name, path=save_loaction)

        # Unzip
        Manager.unzip_dataset(dataset_name, save_loaction)


    @staticmethod
    def unzip_dataset(dataset_name:str, save_loaction:str) -> None:
        zip_path = os.path.join(save_loaction, f"{dataset_name}.zip")

        # Unzip the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_loaction)

        # Remove zip file
        os.remove(zip_path)
