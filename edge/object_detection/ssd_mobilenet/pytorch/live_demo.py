import argparse
import json
import time
import tempfile

import cv2

import torch
import torchvision
import numpy as np

from ssd_mobilenet_v1 import get_tf_pretrained_mobilenet_ssd


def pil_to_tensor(image):
    x = np.asarray(image).astype(np.float32)
    x = torch.as_tensor(x).permute(2, 0, 1)
    return x


import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="model")
    return graph


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--device", default="cpu", help="torch.device to use for inference [cpu, cuda]"
    )
    parser.add_argument(
        "--class-names",
        default="coco-model-labels.txt",
        help="path to pre-trained weights",
    )
    parser.add_argument(
        "--weights-file",
        default="ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
        help="path to pre-trained weights",
    )

    args = parser.parse_args()

    setup_time_begin = time.time()
    device = torch.device(args.device)

    model = get_tf_pretrained_mobilenet_ssd(args.weights_file)
    model.eval()
    model.to(device)

    graph = load_graph(args.weights_file)
    x = graph.get_tensor_by_name('model/image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('model/detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('model/detection_scores:0')
    num_detections = graph.get_tensor_by_name('model/num_detections:0')
    detection_classes = graph.get_tensor_by_name('model/detection_classes:0')

    with open(args.class_names, 'r') as f:
        class_names = f.readlines()
        class_names = [x.strip() for x in class_names]

    cap = cv2.VideoCapture(0)
    with tf.Session(graph=graph) as sess:
      while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            continue
        image0 = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = pil_to_tensor(image0)[None]
        
        # from torch.nn import functional as F
        # i = F.interpolate(image, (300, 300), mode='bilinear', align_corners=True)
        # t = graph.get_tensor_by_name('model/Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0')
        # t = graph.get_tensor_by_name('model/Preprocessor/sub:0')
        # tt = sess.run(t, feed_dict={x: image0[None]})

        # Detect image
        detect_time_begin = time.time()
        image = image.to(device)
        with torch.no_grad():
            boxes, labels, scores = model.predict(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
        d_boxes, d_scores, d_num, d_classes = sess.run((detection_boxes, detection_scores, num_detections, detection_classes), feed_dict={
            x: image0[None]
        })
        # from IPython import embed; embed()
        # sys.sdfsdf
        d_num = int(d_num)
        d_boxes = d_boxes[0, :d_num]
        d_scores = d_scores[0, :d_num].tolist()
        d_classes = d_classes[0, :d_num].astype('int64').tolist()

        height, width = image.shape[-2:]
        d_boxes[:, ::2] *= height
        d_boxes[:, 1::2] *= width
        d_boxes = d_boxes[:, [1, 0, 3, 2]].astype('int64')

        # boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {scores[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        for i in range(d_boxes.shape[0]):
            box = d_boxes[i, :]
            label = f"{class_names[d_classes[i]]}: {d_scores[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255,255), 4)

            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 255, 255),
                        2)  # line type

        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
