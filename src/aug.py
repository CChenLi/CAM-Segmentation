import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

imgs = [
    "data/images/Abyssinian_1.jpg",
    "data/images/Abyssinian_2.jpg",
    "data/images/Abyssinian_3.jpg",
    "data/images/Abyssinian_4.jpg",
    "data/images/Abyssinian_5.jpg",
]

masks = [
    "data/annotations/trimaps/Abyssinian_1.png",
    "data/annotations/trimaps/Abyssinian_2.png",
    "data/annotations/trimaps/Abyssinian_3.png",
    "data/annotations/trimaps/Abyssinian_4.png",
    "data/annotations/trimaps/Abyssinian_5.png",
]

augNtransform = A.Compose([
    A.Resize(128,128),
    A.Rotate(limit=35,p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    # ToTensorV2()  
])

plt.figure(figsize =(20, 8))
plt.ion()
plt.show()
for i in range(5):
    image = cv2.imread(imgs[i])
    mask = cv2.imread(masks[i])
    mask = (mask > 1) * 1.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = augNtransform(image=image, mask = mask)
    timage = transformed['image']
    tmask = transformed['mask']
    print(tmask)

    ax = plt.subplot(2, 5, i+1)
    ax.imshow(timage)
    ax = plt.subplot(2, 5, i+6)
    ax.imshow(tmask)

plt.savefig("augtest.jpg")


