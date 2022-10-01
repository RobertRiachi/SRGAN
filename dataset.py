import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HIGH_RESOLUTION = 256
LOW_RESOLUTION = HIGH_RESOLUTION // 4

class ImageDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__()

        self.image_data = []
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)

        for idx, file_name in enumerate(self.images):
            image_path = os.path.join(self.root_dir, file_name)
            img = Image.open(image_path)
            width, height = img.size

            # Validate that image data isn't smaller than our desired dimensions
            if width < HIGH_RESOLUTION or height < HIGH_RESOLUTION:
                continue

            self.image_data.append(image_path)
        
        print(f"Built Image Dataset using {len(self.image_data)} images")
    
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        file_name = self.image_data[index]
        image = np.array(Image.open(file_name))
        image = preprocess_image(image=image)["image"]
        
        low_resolution_image = process_lowres(image=image)["image"]
        high_resolution_image = process_highres(image=image)["image"]

        return low_resolution_image, high_resolution_image

preprocess_image = A.Compose(
    [
        A.RandomCrop(width=HIGH_RESOLUTION, height=HIGH_RESOLUTION),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ]
)

process_lowres = A.Compose(
    [
        A.Resize(width=LOW_RESOLUTION, height=LOW_RESOLUTION, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2()
    ]
)

process_highres = A.Compose(
    [
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ToTensorV2()
    ]
)


test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

if __name__ == "__main__":
    dataset = ImageDataset("data/train")
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)

    print(dataset[0])

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)
        break