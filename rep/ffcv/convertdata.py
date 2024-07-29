from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
# my_dataset = make_my_dataset()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

my_dataset = datasets.ImageFolder(
    "~/data/test-accuracy/imagenette2/val")
write_path = 'val.beton'

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(
        max_resolution=256
    ),
    'label': IntField()
})

# Write dataset
writer.from_indexed_dataset(my_dataset)