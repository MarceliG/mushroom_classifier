import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocess:

    def __init__(self, mushrooms_path):
        self.mushrooms_path = mushrooms_path

        self.data = {}

    # Lista obrazów
    def get_image(self):
        image_list = []

        for filepath in glob.glob(os.path.join(self.mushrooms_path, "*", "*.jpg"), recursive=True):
            mushroom_class = filepath.split("/")[-2]
            image_list.append((filepath, mushroom_class))
        
        self.data = pd.DataFrame(image_list, columns = ['filepath', 'name'])

    # Pobierz nazwy klas
    def get_classes(self):
        return self.data['name'].unique()

    
    def execute(self):
        self.get_image()
        mushroom_classes = self.get_classes()

        train_data = pd.DataFrame(columns = ['filepath', 'name'])
        val_data = pd.DataFrame(columns = ['filepath', 'name'])
        test_data = pd.DataFrame(columns = ['filepath', 'name'])

        # Tworzę dane treningowe i testowe
        full_train_data = pd.DataFrame(columns = ['filepath', 'name'])
        for mushroom in mushroom_classes:
            temp = self.data[self.data['name'] == mushroom].copy()

            train, test = train_test_split(temp, test_size=0.1, train_size=0.9)

            train_ls = train[['name', 'filepath']]
            test_ls = test[['name', 'filepath']]

            full_train_data = pd.concat([full_train_data, train_ls], ignore_index=True, sort=False)
            test_data = pd.concat([test_data, test_ls], ignore_index=True, sort=False)

        # Tworzę dane treningowe, walidacyjne i testowe
        train_data = pd.DataFrame(columns = ['filepath', 'name'])
        val_data = pd.DataFrame(columns = ['filepath', 'name'])
        for mushroom in mushroom_classes:
            temp = full_train_data[full_train_data['name'] == mushroom].copy()
            train, test = train_test_split(temp, test_size=0.09)
            
            train_ls = train[['name', 'filepath']]
            test_ls = test[['name', 'filepath']]
            
            train_data = pd.concat([train_data, train_ls], ignore_index=True, sort=False)
            val_data = pd.concat([val_data, test_ls], ignore_index=True, sort=False)

        
        return (train_data, val_data, test_data)

        





