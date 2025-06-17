from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor

train_transforms = Compose([
    LoadImage(image_only=True),
    Resize((28, 28)),
    ScaleIntensity(),
    EnsureChannelFirst(),  # dari HWC ke CHW
    ToTensor()
])

val_transforms = Compose([
    LoadImage(image_only=True),
    Resize((28, 28)),
    ScaleIntensity(),
    EnsureChannelFirst(),
    ToTensor()
])