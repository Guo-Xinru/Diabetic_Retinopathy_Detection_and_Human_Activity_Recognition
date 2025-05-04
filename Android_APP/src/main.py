import os
import copy
import math
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from plyer import accelerometer
from plyer import gyroscope
import numpy as np
import kivy
from model import TensorFlowModel


class DemoApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        self.data = []
        self.actions = [
            "WALKING",  # 1
            "WALKING_UPSTAIRS",  # 2
            "WALKING_DOWNSTAIRS",  # 3
            "SITTING",  # 4
            "STANDING",  # 5
            "LAYING",  # 6
            "STAND_TO_SIT",  # 7
            "SIT_TO_STAND",  # 8
            "SIT_TO_LIE",  # 9
            "LIE_TO_SIT",  # 10
            "STAND_TO_LIE",  # 11
            "LIE_TO_STAND"  # 12
        ]
        self.model = TensorFlowModel()
        self.model.load(os.path.join(os.getcwd(), 'model.tflite'))
        """
        Check if the accelerometer and gyroscope can be accessed on device
        """
        try:
            accelerometer.enable()
            print("Accelerometer enabled.")
        except (NotImplementedError, AttributeError) as e:
            print(f"Accelerometer enable failed: {e}")
            layout.add_widget(Label(text="Accelerometer not supported."))
        except Exception as e:
            print(f"Unexpected error enabling accelerometer: {e}")
            layout.add_widget(Label(text="Error enabling accelerometer."))

        try:
            gyroscope.enable()
            print("Gyroscope enabled.")
        except (NotImplementedError, AttributeError) as e:
            print(f"Gyroscope enable failed: {e}")
            layout.add_widget(Label(text="Gyroscope not supported or unavailable."))
        except Exception as e:
            print(f"Unexpected error enabling gyroscope: {e}")
            layout.add_widget(Label(text="Error enabling gyroscope."))

        # Create labels for displaying sensor data
        self.accel_label = Label(text="Accelerometer\nX: 0\nY: 0\nZ: 0", font_size='30sp')
        self.gyro_label = Label(text="Gyroscope\nPitch: 0\nRow: 0\nYaw: 0", font_size='30sp')
        self.predict_label = Label(text="predict action: ", font_size='50sp')

        # Add labels to layout
        layout.add_widget(self.accel_label)
        layout.add_widget(self.gyro_label)
        layout.add_widget(self.predict_label)

        Clock.schedule_interval(self.update_label, 0.02)  # Update every 0.5 second
        return layout

    def update_label(self, dt):
        # Update accelerometer data
        try:
            accel_data = accelerometer.acceleration
            if accel_data:
                ax, ay, az = accel_data
                balanced_az = az - 9.806
                self.accel_label.text = f"Accelerometer\nX: {ax:.2f}\nY: {ay:.2f}\nZ: {balanced_az:.2f}"
            else:
                ax, ay, az = 0.0, 0.0, 0.0
                self.accel_label.text = "Accelerometer\nNo data available."
        except Exception as e:
            ax, ay, az = 0.0, 0.0, 0.0
            self.accel_label.text = f"Accelerometer\nError: {e}"
            print(f"Error in accelerometer: {e}")  # Print detailed error

        # Update gyroscope data
        try:
            gyro_data = gyroscope.orientation
            if gyro_data:
                gx, gy, gz = gyro_data
                self.gyro_label.text = f"Gyroscope\nPitch: {gx:.2f}\nRow: {gy:.2f}\nYaw: {gz:.2f}"
            else:
                gx, gy, gz = 0.0, 0.0, 0.0
                self.gyro_label.text = "Gyroscope\nNo data available."
        except Exception as e:
            gx, gy, gz = 0.0, 0.0, 0.0
            self.gyro_label.text = f"Gyroscope\nError: {e}"
            print(f"Error in gyroscope: {e}")  # Print detailed error

        # append to the be predicted data
        new_element = [ax, ay, balanced_az, gx, gy, gz]
        self.data.append(copy.deepcopy(new_element))
        if len(self.data)==250:
            print(f"data full")
            data_norm = self.normalization()

            data_norm = np.expand_dims(data_norm, axis=0)  # Add batch dimension if needed
            print("Prepared data shape:", data_norm.shape)

            result = self.model.pred(data_norm)
            print(f"Raw Prediction Result: {result}")

            # Get the label index (argmax)
            label_index = np.argmax(result, axis=-1)  # Get index of max probability
            predicted_action = self.actions[label_index[0] - 1]
            self.predict_label.text = f"action:\n {predicted_action}"
            self.data.clear()


    def on_stop(self):
        # Disable sensors, clear the data resource
        Clock.unschedule(self.update_label)
        try:
            accelerometer.disable()
            gyroscope.disable()
            print("Sensors disabled.")
        except Exception as e:
            print(f"Error disabling sensors: {e}")

    def normalization(self):
        # mean
        num_columns = len(self.data[0]) if self.data else 0
        mean_vals = [sum(row[i] for row in self.data) / len(self.data) for i in range(num_columns)]

        # std
        std_vals = []
        for i in range(num_columns):
            mean = mean_vals[i]
            variance = sum((row[i] - mean) ** 2 for row in self.data) / len(self.data)
            std_vals.append(math.sqrt(variance))

        # normalized
        normalized_data = []
        for row in self.data:
            normalized_row = [
                (row[i] - mean_vals[i]) / (std_vals[i] if std_vals[i] != 0 else 1)
                for i in range(num_columns)
            ]
            normalized_data.append(normalized_row)

        # Convert to numpy array
        return np.array(normalized_data, dtype=np.float32)

        #  jnius.jnius.JavaException: JVM exception occurred: Cannot copy to a TensorFlowLite tensor (serving_default_input_layer:0) with 120000 bytes from a Java Buffer with 6000 bytes.

if __name__ == '__main__':
    DemoApp().run()
