from sklearn.metrics import accuracy_score, recall_score
from sklearn import tree
import matplotlib.pyplot as plt
from training import *
from main import display_confusion_matrix


# This function evaluates a model based on accuracy, sensitivity, and specificity.
# It also optionally displays the decision tree and a confusion matrix.
def evaluate_and_display_results(X_test, y_test, predictions, model, show_matrix, title):
    accuracy = accuracy_score(y_test, predictions)
    sensitivity = recall_score(y_test, predictions, pos_label='M')
    specificity = recall_score(y_test, predictions, pos_label='B')
    print(f"{title} - \nAccuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}")
    if model:
        plt.figure(figsize=(12, 8))
        tree.plot_tree(model, filled=True)
        plt.title(f"{title} - Depth: {model.max_depth}")
        plt.show()
    if show_matrix:
        display_confusion_matrix(y_test, predictions, title)

# This function evaluates a combined model based on accuracy, sensitivity, and specificity.
# It returns these metrics without displaying any visualizations.
def evaluate_combined_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    sensitivity = recall_score(y_test, predictions, pos_label='M')
    specificity = recall_score(y_test, predictions, pos_label='B')
    return accuracy, sensitivity, specificity
