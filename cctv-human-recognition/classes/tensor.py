import tensorflow.compat.v1 as tf
import time

class Tensor:
    def __init__(self, path):
        with open('labels.txt', 'r') as f:
            self.__model_labels = [line.strip() for line in f.readlines()]

        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            t1 = time.time()
            fid = tf.gfile.GFile(path, 'rb')
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            t2 = time.time()
            print("model loading time: ", t2 - t1)

            self.__session = tf.Session(graph=detection_graph)
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.__tensor_dict = {}
            keys = [
                'num_detections',
                'detection_boxes',
                'detection_scores',
                'detection_classes'
            ]
            for key in keys:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.__tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            self.__image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    @property
    def tensor_dict(self):
        return self.__tensor_dict

    @property
    def detection_graph_session(self):
        return self.__session

    @property
    def image_tensor(self):
        return self.__image_tensor

    @property
    def model_labels(self):
        return self.__model_labels
