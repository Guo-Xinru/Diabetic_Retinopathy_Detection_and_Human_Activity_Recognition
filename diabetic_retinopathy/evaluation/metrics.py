import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs, num_classes):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # self defined funcktion
        self.num_classes = num_classes


    def update_state(self, *args, **kwargs):
        # ...

    def result(self):
        # ...

    def calculate_CM(self, binary = True, test_dataset, model):
        # define the different class
        for val_images, val_labels in self.test_dataset:
            self.val_step(val_images, val_labels)
            predictions_output = self.model(val_images)
            predictions_output = tf.argmax(predictions_output, axis=1)
            print(f"Predicted: {predictions_output}")

        prediction_labels = []
        true_labels = []

        conf_matrix = tf.math.confusion_matrix(labels=y_true_tensor, predictions=y_pred_tensor, num_classes=3)

        print("Confusion Matrix:")
        print(conf_matrix.numpy())

        # 可视化混淆矩阵（如果需要）
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix.numpy(), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()


