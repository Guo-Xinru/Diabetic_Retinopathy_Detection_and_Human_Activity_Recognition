import numpy as np
from kivy.utils import platform

if platform == 'android':
    from jnius import autoclass

    File = autoclass('java.io.File')
    Interpreter = autoclass('org.tensorflow.lite.Interpreter')
    InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
    Tensor = autoclass('org.tensorflow.lite.Tensor')
    DataType = autoclass('org.tensorflow.lite.DataType')
    ByteBuffer = autoclass('java.nio.ByteBuffer')

    class TensorFlowModel():
        def load(self, model_filename, num_threads=None):
            model = File(model_filename)
            options = InterpreterOptions()
            if num_threads is not None:
                options.setNumThreads(num_threads)
            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()

        def allocate_tensors(self):
            self.interpreter.allocateTensors()
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                self.interpreter.resizeInput(0, shape)
                self.allocate_tensors()

        def pred(self, x):
            # Create a ByteBuffer for the input tensor
            input_buffer = ByteBuffer.allocateDirect(x.nbytes)
            input_buffer.put(x.tobytes())
            input_buffer.rewind()

            # Manually create an output ByteBuffer with the appropriate size
            output_size = 1
            for dim in self.output_shape:
                output_size *= dim

            if self.output_type == DataType.FLOAT32:
                output_buffer = ByteBuffer.allocateDirect(output_size * 4)  # 4 bytes per float
            elif self.output_type == DataType.UINT8:
                output_buffer = ByteBuffer.allocateDirect(output_size)  # 1 byte per uint8
            else:
                raise ValueError("Unsupported output type")

            output_buffer.rewind()

            # Run the interpreter
            self.interpreter.run(input_buffer, output_buffer)

            # Convert the output ByteBuffer to a numpy array
            output_array = np.zeros(self.output_shape,
                                    dtype=np.float32 if self.output_type == DataType.FLOAT32 else np.uint8)
            output_buffer.rewind()

            if self.output_type == DataType.FLOAT32:
                for i in range(output_array.size):
                    output_array.flat[i] = output_buffer.getFloat()
            elif self.output_type == DataType.UINT8:
                for i in range(output_array.size):
                    output_array.flat[i] = output_buffer.get() & 0xFF
            print(output_array)
            return np.reshape(output_array, self.output_shape)

elif platform == 'ios':
    from pyobjus import autoclass, objc_arr
    from ctypes import c_float, cast, POINTER

    NSString = autoclass('NSString')
    NSError = autoclass('NSError')
    Interpreter = autoclass('TFLInterpreter')
    InterpreterOptions = autoclass('TFLInterpreterOptions')
    NSData = autoclass('NSData')
    NSMutableArray = autoclass("NSMutableArray")

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None):
            self.error = NSError.alloc()
            model = NSString.stringWithUTF8String_(model_filename)
            options = InterpreterOptions.alloc().init()
            if num_threads is not None:
                options.numberOfThreads = num_threads
            self.interpreter = Interpreter.alloc(
            ).initWithModelPath_options_error_(model, options, self.error)
            self.allocate_tensors()

        def allocate_tensors(self):
            self.interpreter.allocateTensorsWithError_(self.error)
            self.input_shape = self.interpreter.inputTensorAtIndex_error_(
                0, self.error).shapeWithError_(self.error)
            self.input_shape = [
                self.input_shape.objectAtIndex_(_).intValue()
                for _ in range(self.input_shape.count())
            ]
            self.output_shape = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).shapeWithError_(self.error)
            self.output_shape = [
                self.output_shape.objectAtIndex_(_).intValue()
                for _ in range(self.output_shape.count())
            ]
            self.output_type = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).dataType

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                # workaround as objc_arr doesn't work as expected on iPhone
                array = NSMutableArray.new()
                for x in shape:
                    array.addObject_(x)
                self.interpreter.resizeInputTensorAtIndex_toShape_error_(
                    0, array, self.error)
                self.allocate_tensors()

        def pred(self, x):
            # assumes one input and one output for now
            bytestr = x.tobytes()
            # must cast to ctype._SimpleCData so that pyobjus passes pointer
            floatbuf = cast(bytestr, POINTER(c_float)).contents
            data = NSData.dataWithBytes_length_(floatbuf, len(bytestr))
            print(dir(self.interpreter))
            self.interpreter.copyData_toInputTensor_error_(
                data, self.interpreter.inputTensorAtIndex_error_(
                    0, self.error), self.error)
            self.interpreter.invokeWithError_(self.error)
            output = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).dataWithError_(self.error).bytes()
            # have to do this to avoid memory leaks...
            while data.retainCount() > 1:
                data.release()
            return np.reshape(
                np.frombuffer(
                    (c_float * np.prod(self.output_shape)).from_address(
                        output.arg_ref), c_float), self.output_shape)

else:
    import tensorflow as tf

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None):
            self.interpreter = tf.lite.Interpreter(model_filename,
                                                   num_threads=num_threads)
            self.interpreter.allocate_tensors()

        def resize_input(self, shape):
            if list(self.get_input_shape()) != shape:
                self.interpreter.resize_tensor_input(0, shape)
                self.interpreter.allocate_tensors()

        def get_input_shape(self):
            return self.interpreter.get_input_details()[0]['shape']

        def pred(self, x):
            # assumes one input and one output for now
            self.interpreter.set_tensor(
                self.interpreter.get_input_details()[0]['index'], x)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(
                self.interpreter.get_output_details()[0]['index'])