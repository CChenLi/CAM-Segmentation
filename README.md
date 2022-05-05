# CAM-Segmentation
Project for COMS 4995 DeepLearning S22: Class activation map-based image segmentaion.

### Motivation
- The underlying task behind two class segmentation highly overlay with image classification.
- The task for segmentation model is to separate the main object from the background of the image. However, there can be many category of main object (e.g. cat, dog, flower). Segmentation model only receives pixel-level label, which can't provide sufficient information of learn a seperatble distribution across different classes.
- Incorperating a pretrained classifier's feature extractor to capture the seperatble distribution of objects across different classes, which help the model to have a holistic understanding of the target object to enhance segmentation accuracy.



### File dependencies
- Most Model related works are shown in `CAMUnet.ipynb`
- Sementation models are defined in `src/unet.py`
- Classifier is defined in `src/Rescnn.py`
- The dataset in defined in `src/pet_dataset.py`
- `Trainer` class defined in `src/train.py` summarize all the experiment setups. experiment name includes 
  - *base_model*, which set up the experiment for plain UNet
  - *cam_model*, which set up the experiment for CAM-UNet
  - *clf_model*, which set up the experiment for Classifier-UNet

> The implementation of plain UNet is based on [UNet](https://github.com/milesial/Pytorch-UNet#training)
