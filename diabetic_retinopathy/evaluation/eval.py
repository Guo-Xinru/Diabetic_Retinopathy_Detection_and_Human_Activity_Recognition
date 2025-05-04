import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    """
        Evaluate a model using a given test dataset and checkpoint.

        Parameters:
            model (tf.keras.Model): The model to evaluate.
            checkpoint (str): Path to checkpoint
            ds_test (tf.data.Dataset): The test dataset.
            ds_info (dict): Dataset information (e.g., class names, image shape).
            run_paths (dict): Paths for saving logs, results, etc.

        Returns:
            dict: Evaluation results including accuracy and loss.
        """

    # load the checkpoint
    #reader = tf.train.load_checkpoint(checkpoint)
    #result = model.evaluate(ds_test, ds_info)

    print(f"Loading checkpoint from {checkpoint}...")
    checkpoint_obj = tf.train.Checkpoint(model=model)
    checkpoint_obj.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()
    print("Checkpoint restored successfully.")

    # Compile the model (if metrics are needed)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.load_weights(checkpoint)

    # Evaluate the model on the test dataset
    print("Evaluating the model...")
    results = model.evaluate(ds_test, return_dict=True)

    #calculate the Confusion Matrix
    y_true = []
    y_pred = []
    for images, labels in ds_test:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    conf_matrix = confusion_matrix(y_true, y_pred)
    # class_report = classification_report(y_true, y_pred, target_names=ds_test.get('class_names', None))
    # print(y_true)
    # print(y_pred)
    cm_visu(conf_matrix, test_dataset=ds_test, save_path=os.getcwd() + '/visusalization/')


    # Optionally log and save results
    print(f"Test Loss: {results['loss']:.4f}, Test Accuracy: {results['accuracy']:.2%}")
    results_path = f"{run_paths['path_model_id']}/evaluation_results.json"
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")

    # calculate the recall, precision, F1 Score , Sensitivity
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')


    # Compute Sensitivity (for binary classification)
    # Sensitivity is equivalent to Recall for the positive class
    conf_matrix = confusion_matrix(y_true, y_pred)
    if len(conf_matrix) == 2:  # Binary classification case
        TP = conf_matrix[1, 1]
        FN = conf_matrix[1, 0]
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    else:
        sensitivity = None  # Sensitivity is not defined for multi-class directly

    # Compute Specificity
    specificity_list = []
    """
    for i in range(len(conf_matrix)):
        tn = sum(sum(conf_matrix)) - (sum(conf_matrix[i, :]) + sum(conf_matrix[:, i]) - conf_matrix[i, i])
        fp = sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_list.append(specificity)
    specificity = sum(specificity_list) / len(specificity_list)
    """
    for i in range(len(conf_matrix)):
        fp = sum(sum(conf_matrix)) - (sum(conf_matrix[i, :]) + sum(conf_matrix[:, i]) - conf_matrix[i, i])
        tn = sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = fp / (fp + tn) if (tn + fp) > 0 else 0.0
        specificity_list.append(specificity)
    specificity = sum(specificity_list) / len(specificity_list)

    print('Precision', precision)
    print('Recall', recall)
    print('F1Score', f1)
    print('Sensitivity', sensitivity)
    print('Specificity', specificity)

    return results



"""
Inference the NN from the trained Model
"""
def inference(model, checkpoint, ds_test):

    # load the weight from the checkpoint
    model.load_weights(checkpoint)

    y_true = []
    y_pred = []
    for images, labels in ds_test:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    print(y_true)
    print(y_pred)



"""
visualisation the confusion matrix
"""
def cm_visu(confusion_matrix, test_dataset, save_path):
    # Visualize confusion matrix
    print("Visualizing confusion matrix...")
    class_names = [0,1,2,3,4]
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = confusion_matrix.max() / 2.0
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    conf_matrix_fig_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(conf_matrix_fig_path)
    print(f"Confusion matrix visualization saved to {conf_matrix_fig_path}")
    plt.close()