from data_reader import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from training import *
from eval import *

#This function calls and prints results for functions according to project specifications. 
def main():
    print("---------------------- QUESTION 1 ----------------------")
    print("Reading in data...")
    data = get_data()

    print("---------------------- QUESTION 2 ----------------------")
    print("Clearing data...")
    print("Splitting dataset [80/20]...")

    print("---------------------- QUESTION 3 ----------------------")
    training_time_q3 = train_and_evaluate_single_model(data, DecisionTreeClassifier, True, "Decision Tree all features")

    print("---------------------- QUESTION 4 ----------------------")
    ensemble_decision_trees(data, training_time_q3)

    print("---------------------- QUESTION 5 ----------------------")
    train_svm_and_visualize(data)

    print("---------------------- QUESTION 6 ----------------------")
    train_svm_ensemble(data)

# This function combines predictions from multiple models into a single prediction.
# It takes a list of predictions as input and returns a list of final labels.
def combine_predictions(pred_list):
    label_mapping = {'B': 0, 'M': 1}
    int_predictions = [[label_mapping[label] for label in pred] for pred in pred_list]
    combined_pred = np.stack(int_predictions)
    final_pred = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=combined_pred)
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    final_labels = [inv_label_mapping[i] for i in final_pred]
    return final_labels

# This function plots decision boundaries for a given model on a 2D feature space.
# It visualizes the separation between different classes predicted by the model.
def plot_decision_boundaries(X, y, model, title_suffix):
    if y.dtype.kind in 'UO':  
        unique_labels = np.unique(y)
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_to_int[label] for label in y])

    h = .02  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    if np.any([isinstance(z, str) for z in np.unique(Z)]):
        Z = np.array([label_to_int[z] for z in Z])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm if 'RBF' in title_suffix else plt.cm.viridis, alpha=0.8)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm if 'RBF' in title_suffix else plt.cm.viridis, edgecolors='k')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f"{title_suffix} - Decision Boundary")
    plt.show()

# This function displays a confusion matrix for given predictions and actual labels.
# It visually represents the performance of a classification model.
def display_confusion_matrix(y_test, predictions, title):
    confusion = confusion_matrix(y_test, predictions)
    cm_display = ConfusionMatrixDisplay(confusion, display_labels=["B", "M"])
    cm_display.plot()
    plt.title(title + " - Confusion Matrix")
    plt.show()

# This function compares training times for different decision tree models.
# It visualizes the trees and their respective training times.
def compare_training_times(models, training_times, training_time_q3):
    print("Training Time Comparison:")
    print(f"Question 3 Training Time: {training_time_q3:.2f}s")
    for model, training_time in zip(models, training_times):
        plt.figure(figsize=(12, 8))
        tree.plot_tree(model, filled=True)
        plt.title(f"Decision Tree (Depth {model.max_depth}) - Training Time: {training_time:.2f}s")
        plt.show()
    print("Question 4 Training Times: " + ", ".join([f"{time:.2f}s" for time in training_times]))

# This function compares training times for different SVM models.
# It prints a comparison of the training times for different SVM models.
def compare_svm_training_times(training_times, previous_time):
    print("SVM Training Time Comparison:")
    print(f"Question 5 Training Time: {previous_time:.2f}s")
    print("Question 6 SVM Training Times: " + ", ".join([f"{time:.2f}s" for time in training_times]))

if __name__ == '__main__':
    main()