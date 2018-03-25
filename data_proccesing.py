import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

class DataProcessing: 
    df = ""
    def processing(self, file, column_amount):

        columns = self.create_table_label(column_amount)
        df = pd.read_csv(file)
        df = pd.DataFrame(df.values.tolist(), columns=columns)

        selected_columns = [3, 2, 4, 2, 13, 2, 4, 13, 6, 13, 6, 12]
        selected_columns = set(selected_columns)
        selected_columns.add('Class')

        df = df.astype('float32')

        labels = df.iloc[:, 0].values
        features = df[list(selected_columns)].values

        X_train, X_test = self.split_date(features)
        Y_train, Y_test = self.split_date(labels)

        self.save_np(X_train, Y_train, X_test, Y_test)

        return df 

    def create_table_label(self, attribute_amount):
        columns = [i - 1 for i in range(1,attribute_amount + 1)]
        columns[0] = "Class"
        return columns
    
    # TODO select heat map with or without class
    def plot_heat_map(self, size, values):
        plt.figure(figsize=(size, size))
        return sns.heatmap(values.iloc[:, 1:].corr(), annot=True)

    def split_date(self, data, percent=0.8):
        index = int(percent * len(data))
        return data[:index], data[index:]

    def save_np(self, X_train, Y_train, X_test, Y_test):
        np.save('dataset/X_train', X_train)
        np.save('dataset/Y_train', Y_train)
        np.save('dataset/X_test', X_test)
        np.save('dataset/Y_test', Y_test)
        
    