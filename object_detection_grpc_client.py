import argparse
import os
import time
import sys
import tensorflow as tf
from PIL import Image
import numpy as np
from grpc.beta import implementations
sys.path.append("..")
from object_detection.core.standard_fields import \
    DetectionResultFields as dt_fields
from object_detection.utils import label_map_util
from argparse import RawTextHelpFormatter
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.logging.set_verbosity(tf.logging.INFO)

WIDTH = 1024
HEIGHT = 768


def load_image_into_numpy_array(input_image):
    image = Image.open(input_image)
    image = image.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    (im_width, im_height) = image.size
    image_arr = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    image.close()
    return image_arr


def load_input_tensor(input_image):
    start_array_numpy = time.time()

    image_np = load_image_into_numpy_array(input_image)
    end_array_numpy = time.time()
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.contrib.util.make_tensor_proto(image_np_expanded)
    print("numpy array" + str(end_array_numpy - start_array_numpy))
    return tensor


def main(args):
    start_main = time.time()

    host, port = args.server.split(':')

    channel = implementations.insecure_channel(host, int(port))._channel

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.model_name

    input_tensor = load_input_tensor(args.input_image)
    print(str(input_tensor)[0:100])
    request.inputs['inputs'].CopyFrom(input_tensor)
    # print(str(request)[0:100])
    start = time.time()

    result = stub.Predict(request, 60.0)
    end = time.time()

    # image_np = load_image_into_numpy_array(args.input_image)

    output_dict = {}
    # print(result.outputs[dt_fields.detection_classes])
    output_dict[dt_fields.detection_classes] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8)
    output_dict[dt_fields.detection_boxes] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict[dt_fields.detection_scores] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)
    category_index = label_map_util.create_category_index_from_labelmap(args.label_map,
                                                                        use_display_name=True)
    classes = output_dict[dt_fields.detection_classes]
    scores = output_dict[dt_fields.detection_scores]
    classes.shape = (1, 300)
    scores.shape = (1, 300)
    print("prediction time : " + str(end-start))
    objects = []
    threshold = 0.5  # in order to get higher percentages you need to lower this number; usually at 0.01 you get 100% predicted objects
    for index, value in enumerate(classes[0]):
        object_dict = {}
        if scores[0, index] > threshold:
            object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                scores[0, index]
            objects.append(object_dict)
    print(objects)
    end_main = time.time()

    print("end_main" + str(end_main-start_main))

    # output_img.save(output_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object detection grpc client.",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--server',
                        type=str,
                        default='localhost:9000',
                        help='PredictionService host:port')
    parser.add_argument('--model_name',
                        type=str,
                        default="aadhaarmodel",
                        help='Name of the model')
    parser.add_argument('--input_image',
                        type=str,
                        default='./test_images/123.jpg',
                        help='Path to input image')
    parser.add_argument('--output_directory',
                        type=str,
                        default='./',
                        help='Path to output directory')
    parser.add_argument('--label_map',
                        type=str,
                        default="./data/object_detection.pbtxt",
                        help='Path to label map file')

    args = parser.parse_args()
    main(args)
