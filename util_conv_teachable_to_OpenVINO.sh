#!/bin/bash

# if not already present - install with sudo apt-get install unzip
echo "---0. Pulling TF and OpenVINO images--"
unzip converted_keras.zip

TF_CONV_CONTAINER_NAME="tf-temp-conv"
H5_MODEL_FULL_PATH='./keras_model.h5'
SAVE_PB_NAME='frozen_model.pb'
OpenVINO_CONTAINER_NAME="openvino-temp-conv"

echo "---1. Pulling TF and OpenVINO images--"
docker pull tensorflow/tensorflow:1.15.0 &&
docker pull openvino/ubuntu18_dev:latest &&

echo "---2. Creating tf v1.15.0 container ${TF_CONV_CONTAINER_NAME} to convert teachable machine keras h5 to frozen pb--"
docker run -d -t --name $TF_CONV_CONTAINER_NAME -v ${PWD}:/workdir tensorflow/tensorflow:1.15.0 bash &&
docker exec -it $TF_CONV_CONTAINER_NAME bash -c "cd /workdir;python import_h5_export_pb.py ${H5_MODEL_FULL_PATH} ${SAVE_PB_NAME}" &&
echo "---3. Successfully written frozen model ${SAVE_PB_NAME}--"

echo "---4. Stopping ${TF_CONV_CONTAINER_NAME}--"
docker stop $TF_CONV_CONTAINER_NAME > /dev/null &&

echo "---5. Removing ${TF_CONV_CONTAINER_NAME}--"
docker rm $TF_CONV_CONTAINER_NAME > /dev/null

echo "---6. Creating openvino container ${OpenVINO_CONTAINER_NAME} to convert frozen pb to OpenVINO IR --"
docker run -d -t --name $OpenVINO_CONTAINER_NAME -v ${PWD}:/workdir openvino/ubuntu18_dev:latest /bin/bash &&
docker exec -it $OpenVINO_CONTAINER_NAME /bin/bash -c "cd /workdir;python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ${SAVE_PB_NAME} --input_shape [1,224,224,3]" &&
echo "---7. Successfully written frozen model .bin, .xml, .mapping files--"

echo "---8. Stopping ${OpenVINO_CONTAINER_NAME}--"
docker stop $OpenVINO_CONTAINER_NAME > /dev/null &&

echo "---9. Removing ${OpenVINO_CONTAINER_NAME}--"
docker rm $OpenVINO_CONTAINER_NAME > /dev/null
