from numpy import load, array, concatenate, arange, ones, in1d
from torch import tensor, from_numpy, int32, float32, zeros, nonzero
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100, MNIST
from torchvision.transforms import Normalize, RandomHorizontalFlip, RandomCrop, RandomRotation, Compose, ToTensor

# ------------------------------------------------

def load_imagenet(path,train_images_per_class,test_images_per_class,classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []

    for idx, _class in enumerate(classes):
        new_x = load(path + 'classes/' + str(_class) + '.npy')

        x_train.append(new_x[:train_images_per_class])
        y_train.append(array([idx] * train_images_per_class))
        x_test.append(new_x[train_images_per_class:])    
        y_test.append(array([idx] * test_images_per_class))

    x_train = tensor(concatenate(x_train), dtype=float32)
    x_test  = tensor(concatenate(x_test), dtype=float32)
    y_train = from_numpy(concatenate(y_train))
    y_test  = from_numpy(concatenate(y_test))

    return x_train, y_train, x_test, y_test

# ------------------------------------------------

class CifarDataSet(Dataset):

    def __init__(self, path, train):
        # Load data -
        self.classes = arange(100)
        raw_data_set = CIFAR100(root=path, train=train)
        self.data = {"data": raw_data_set.data, "labels": raw_data_set.targets}
        self.transform = None

        # Processing data -
        self.data["data"] = tensor(self.data["data"], dtype=float32).permute((0,3,1,2))/255
        self.integer_labels = self.data["labels"]
        num_samples = len(self.integer_labels)
        one_hot_labels = zeros((num_samples, 100), dtype=float32)
        one_hot_labels[arange(num_samples), self.integer_labels] = 1
        self.data["labels"] = one_hot_labels

        # Task data -
        self.current_data = self.partition_data()

    def __len__(self):
        return self.current_data['data'].shape[0]

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        image = self.current_data['data'][idx]
        label = self.current_data['labels'][idx]

        if self.transform: image = self.transform(image)
        sample = {'image': image, 'label': label}

        return sample

    def partition_data(self):
        current_data = {}

        correct_rows = ones(self.data['data'].shape[0], dtype=bool)
        correct_rows = in1d(self.integer_labels, self.classes)

        current_data['data'] = self.data['data'][correct_rows, :, :, :]
        current_data['labels'] = self.data['labels'][correct_rows][:,self.classes]

        return current_data

    def select_new_partition(self, new_classes):
        self.classes = array(new_classes) #dtype=np.int32)
        self.current_data = self.partition_data()

    def set_transformation(self, new_transformation):
        self.transform = new_transformation

# ------------------------------------------------

def load_cifar(path, train, validation):

    cifar_data = CifarDataSet(path,train)
    transformations = [Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))]

    if not validation:
        transformations.append(RandomHorizontalFlip(p=0.5))
        transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
        transformations.append(RandomRotation(degrees=(0,15)))

    cifar_data.transform = Compose(transformations)

    if not train: return cifar_data

    valid_indices = zeros(50*100, dtype=int32)
    train_indices = zeros(450*100, dtype=int32)
    current_val_samples = 0
    current_train_samples = 0

    for i in range(100):
        class_indices = nonzero(cifar_data.data["labels"][:, i] == 1).flatten()
        valid_indices[current_val_samples:(current_val_samples + 50)] += class_indices[:50]
        train_indices[current_train_samples:(current_train_samples + 450)] += class_indices[50:]
        current_val_samples += 50
        current_train_samples += 450

    indices = valid_indices if validation else train_indices
    cifar_data.data["data"] = cifar_data.data["data"][indices.numpy()]       # .numpy wasn't necessary with torch 2.0
    cifar_data.data["labels"] = cifar_data.data["labels"][indices.numpy()]
    cifar_data.integer_labels = tensor(cifar_data.integer_labels)[indices.numpy()].tolist()
    cifar_data.current_data = cifar_data.partition_data()

    return cifar_data

# ------------------------------------------------

def load_mnist(path):
    train_dataset = MNIST(root=path, train=True, transform=ToTensor(), download=True)
    test_dataset  = MNIST(root=path, train=False, transform=ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=60000, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=60000, shuffle=False)

    for i, (images_train, labels_train) in enumerate(train_loader):
        images_train = images_train.flatten(start_dim=1)
        labels_train = labels_train

    for i, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.flatten(start_dim=1)
        labels_test = labels_test

    return images_train, labels_train, images_test, labels_test