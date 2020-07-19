#!/usr/bin/env python
from openvino.inference_engine import IECore
import numpy as np
import cv2, datetime, sys

if(len(sys.argv) != 5):
    print("teachable_img_openvino_classify.py /path/frozen_model.xml /path/frozen_model.bin /path/to/labels.txt /path/testImage.jpg")

# OpenVINO Plugin Initialization
ie = IECore()

# Read IR
net = ie.read_network(model=sys.argv[1], weights=sys.argv[2])

# Prepare blobs
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

# Read and pre-process input images
n, c, h, w = net.inputs[input_blob].shape
image = cv2.imread(sys.argv[4])
if image.shape[:-1] != (h, w):
    print("Test image resizing to fit...")
    image = cv2.resize(image, (w,h))

image = (np.asarray(image).astype(np.float32)/127.0) - 1 # convert to numpy arrray and normalize the image
image = image.transpose((2, 0, 1)) # change layout from HWC to CHW

# Loading model to the plugin
exec_net = ie.load_network(network=net, device_name="CPU")

# Start sync inference
start_time = datetime.datetime.now()
res = exec_net.infer(inputs={input_blob: image})
end_time = datetime.datetime.now()

# Process output blob and pretty display
res = res[out_blob]
with open(sys.argv[3], 'r') as f:
    labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]

classid_str = "ClassID"
probability_str = "Probability"
for i, probs in enumerate(res):
    probs = np.squeeze(probs)
    top_ind = np.argsort(probs)[-10:][::-1]
    print(classid_str, probability_str)
    print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
    for id in top_ind:
        det_label = labels_map[id] if labels_map else "{}".format(id)
        label_length = len(det_label)
        space_num_before = (len(classid_str) - label_length) // 2
        space_num_after = len(classid_str) - (space_num_before + label_length) + 2
        space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
        print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))

print('Processing time: {:.2f} ms'.format(round((end_time - start_time).total_seconds() * 1000), 2))