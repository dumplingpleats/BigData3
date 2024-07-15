import pandas as pd
from sklearn.model_selection import train_test_split

breast_cancer_df = pd.read_csv("./breast-cancer.csv") # reading

clean_df = breast_cancer_df.dropna() # data cleaning 

# This function reads the breast cancer dataset and performs basic data cleaning.
# It then splits the data into training and testing sets and returns a dictionary containing these sets.
def get_data(columns_to_drop:list = []) -> dict[str, list]:
    new_df = clean_df.drop(columns_to_drop, axis=1)

    X = new_df.iloc[:,2:] # columns with all features
    y = new_df.iloc[:,1] # diagnosis column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data 80/20

    features_list = X.columns.tolist()

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "features_list": features_list, "X_dataframe" :X, "y_dataframe": y}