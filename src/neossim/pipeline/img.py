import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from neossim import utils

log = logging.getLogger(__name__)


class ImageNormalizer:
    """

    """
    def __init__(self, size):
        self._std = None
        self._mean = None
        self._step = None
        self._size = size

    def __call__(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def fit(self, dataset: Dataset, batch_size=128, n_workers=10):
        log.info(f"Fitting ImageNormalizer")

        raw_pipeline = self.create_raw_pipeline(self._size)
        utils.set_transformer(dataset, raw_pipeline)

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)
        self._std, self._mean = get_dataset_mean_std(loader)
        self._step = T.Normalize(mean=self._mean, std=self._std)

    @staticmethod
    def create_raw_pipeline(size):
        return T.Compose([
            T.Resize(size=size),
            T.ToTensor(),
        ])


def get_dataset_mean_std(data_loader):
    """
    Calculate per channel mean and std for all images from the given data_loader.
    Used for normalization of images

    :param data_loader: loader for dataset
    :return: mean and div of dataset

    :see Pytorch-Forum: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
    """
    log.info("Calulating dataset mean and std ...")
    mean = 0.
    std = 0.
    log.info(f"Batches: {len(data_loader)}")
    for images, _ in data_loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(data_loader.dataset)
    std /= len(data_loader.dataset)
    mean = mean.numpy().tolist()
    std = std.numpy().tolist()
    log.info(f"Mean: {mean} std: {std}")
    return mean, std