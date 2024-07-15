from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time
import multiprocessing
from eval import *
from main import combine_predictions, compare_svm_training_times, plot_decision_boundaries, compare_training_times

# This function trains a decision tree model in a separate process.
# It calculates training time and returns the model, predictions, and training time.
def decision_tree_worker(X_train, y_train, X_test, max_depth, queue):
    start_time = time.perf_counter()  # Start timing
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    training_time = time.perf_counter() - start_time
    queue.put((model, predictions, training_time))

# This function trains and evaluates a single model, displaying results and returning the training time.
def train_and_evaluate_single_model(data, classifier, show_matrix: bool, title: str):
    start_time = time.perf_counter()
    model = classifier()
    model.fit(data['X_train'], data['y_train'])
    training_time = time.perf_counter() - start_time
    predictions = model.predict(data['X_test'])
    evaluate_and_display_results(data['X_test'], data['y_test'], predictions, model, show_matrix, title)
    return training_time

# This function trains an ensemble of decision trees using multiprocessing.
# It combines predictions from multiple trees and displays the results.
def ensemble_decision_trees(data, training_time_q3):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    queue = multiprocessing.Queue()
    depths = [3, 5, 7]
    processes = []

    for depth in depths:
        p = multiprocessing.Process(target=decision_tree_worker, args=(X_train, y_train, X_test, depth, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = [queue.get() for _ in processes]
    models, predictions, training_times = zip(*results)
    final_predictions = combine_predictions(predictions)
    accuracy, sensitivity, specificity = evaluate_combined_model(y_test, final_predictions)
    compare_training_times(models, training_times, training_time_q3)
    display_confusion_matrix(y_test, final_predictions, "Combined Decision Trees")
    print(f"Combined Decision Trees - Accuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}")

# This function trains an SVM model with an RBF kernel and visualizes its decision boundary.
# It returns the training time for later comparison.
def train_svm_and_visualize(data):
    X_train, X_test = data['X_train'].iloc[:, :2].to_numpy(), data['X_test'].iloc[:, :2].to_numpy()
    y_train, y_test = data['y_train'], data['y_test']
    start_time = time.time()
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    predictions = model.predict(X_test)
    accuracy, sensitivity, specificity = evaluate_combined_model(y_test, predictions)
    print(f"SVM RBF - \nAccuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nTraining Time: {training_time:.2f}s")
    plot_decision_boundaries(X_train, y_train, model, "SVM RBF")
    display_confusion_matrix(y_test, predictions, "SVM RBF")
    return training_time 

# This function trains an SVM model with a specified kernel in a separate process.
# It returns the model, predictions, and training time.
def svm_worker(X_train, y_train, X_test, kernel, queue):
    start_time = time.perf_counter() 
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    training_time = time.perf_counter() - start_time 
    queue.put((model, predictions, training_time))

# This function trains an ensemble of SVM models with different kernels using multiprocessing.
# It combines predictions and displays the results along with training time comparisons.
def train_svm_ensemble(data):
    X_train, X_test, y_train, y_test = data['X_train'].iloc[:, :2].to_numpy(), data['X_test'].iloc[:, :2].to_numpy(), data['y_train'], data['y_test']
    queue = multiprocessing.Queue()
    kernels = ['linear', 'rbf', 'poly']
    processes = []

    for kernel in kernels:
        p = multiprocessing.Process(target=svm_worker, args=(X_train, y_train, X_test, kernel, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = [queue.get() for _ in processes]
    models, predictions, training_times = zip(*results)
    final_predictions = combine_predictions(predictions)
    accuracy, sensitivity, specificity = evaluate_combined_model(y_test, final_predictions)

    for model, kernel, training_time in zip(models, kernels, training_times):
        plot_decision_boundaries(data['X_train'].iloc[:, :2].to_numpy(), data['y_train'], model, f"SVM {kernel.upper()}")

    display_confusion_matrix(y_test, final_predictions, "SVM Ensemble")
    print(f"SVM Ensemble - Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}, Training Times: {[f'{time:.2f}s' for time in training_times]}")
    compare_svm_training_times(training_times, data.get('svm_training_time', 0))