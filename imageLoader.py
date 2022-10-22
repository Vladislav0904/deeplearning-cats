from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

dataset = ImageFolder("data")
trainData, testData, trainLabel, testLabel = train_test_split(dataset.imgs, dataset.targets, test_size=0.2,
                                                              random_state=0)

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkChannel(dataset)
        self.transform = transform

    def checkChannel(self, dataset):
        datasetRGB = []
        for index in range(len(dataset)):
            if Image.open(dataset[index][0]).getbands() == ('R', 'G', 'B'):
                datasetRGB.append(dataset[index])
        return datasetRGB

    def getResizedImage(self, item):
        image = Image.open(self.dataset[item][0])
        _, _, width, height = image.getbbox()
        if width > height:
            factor = (0, 0, width, height)
        else:
            factor = (0, 0, height, width)
        return image.crop(factor)

    def __getitem__(self, item):
        image = self.getResizedImage(item)
        if transform is not None:
            return self.transform(image), self.dataset[item][1]
        return image, self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)