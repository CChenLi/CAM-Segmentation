# CAM-Segmentation
Project for COMS 4995 DeepLearning S22: Class activation map-based image segmentaion

- Most Model related works are shown in `CAMUnet.ipynb`
- Sementation models are defined in `src/unet.py`
- Classifier is defined in `src/Rescnn.py`
- The dataset in defined in `src/pet_dataset.py`
- `Trainer` class defined in `src/train.py` summarize all the experiment setups. experiment name includes 
  - *base_model*, which set up the experiment for plain UNet
  - *cam_model*, which set up the experiment for CAM-UNet
  - *clf_model*, which set up the experiment for Classifier-UNet

> The implementation of plain UNet is based on [UNet](https://github.com/milesial/Pytorch-UNet#training)
