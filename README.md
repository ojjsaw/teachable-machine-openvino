# Teach a model to classify images and optimize inference with OpenVINO in 20mins with zero setup

## Basic Requirements
- Latest stable Docker installed on a Linux System
- Network connectivity to hub.docker.com (instructions assume no proxy)
- 6th to 10th generation Intel Core or Intel Xeon processors or 3rd generation Intel® Xeon® Scalable processors

## Train and export a custom model in your browser
- Navigate to https://teachablemachine.withgoogle.com/train and start the "Image Project"
- Edit Class Names to labels (add as many), record using the webcam or file upload
- Hit **Train Model** (don't switch tabs while training).. for more info watch the official guides from the menu on the left (Gather Samples and Train your model sections)
- On the *Preview panel*, hit **Export Model**
- On the *Export your model to use it projects*, click the second tab **Tensorflow**
- With *Keras* already selected, click the **Download my model**
- Takes some time! Once done, a file named **converted_keras.zip** will be downloaded.

## Convert any Teachable Machine 2.0 keras model to OpenVINO IR format
- Replace the downloaded converted_keras.zip file in the current directory (alongside this README.md)
- Make sure to have docker and unzip installed on your system
- Run below script:
    - Pulls necessary docker images (openvino/ubuntu18_dev:latest, tensorflow/tensorflow:1.15.0)
    - Extracts the zip file (keras_model.h5, labels.txt)
    - Converts .h5 to frozen .pb (frozen_model.pb)
    - Converts .pb to OpenVINO IR (frozen_model.xml, frozen_model.bin, frozen_model.mapping)
```
./util_conv_teachable_to_OpenVINO.sh
```
(Optional) View/Edit the script vars for custom file name changes.

## Run OpenVINO Inference on a single image based on teachable machine converted model
Make sure to capture your test images pertaining to your custom trained model.
```
docker run --rm -it -v ${PWD}:/workdir openvino/ubuntu18_dev:latest /bin/bash
cd /workdir
python3 teachable_img_openvino_classify.py frozen_model.xml frozen_model.bin labels.txt test.jpg
```
Note: The python script is only **30 lines** of actual code to run cpu optimized inference of a teachable machine trained image classification model. Won't require any code changes for any custom Teachable Machine 2.0 Image Projects

## Run OpenVINO Inference on a webcam
```
xhost +
docker run --rm -it --privileged -v ${PWD}:/workdir -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/video0:/dev/video0 openvino/ubuntu18_dev:latest /bin/bash
cd /workdir
python3 teachable_livecam_openvino_classify.py frozen_model.xml frozen_model.bin labels.txt
```

## Run the original keras example script provided on teachable machine website for performance comparsion
```
docker run --rm -it -v ${PWD}:/workdir tensorflow/tensorflow:1.15.0 bash
cd /workdir
pip install Pillow
python teachable_img_keras_orig_classify.py keras_model.h5 test.jpg
```

## Known Limitations
- Only Teachable Machine: Image projects are covered for now.
- Depending upon popularity, Audio and Pose projects can be covered in the future.


