from torchvision.datasets import STL10

# Download the STL10 dataset
# This file is not needed to be run. The data will be downloaded automatically on the fly if not present.
# It can however be useful for pre-downloading the dataset ahead of time.
directory = "data/STL10"
STL10(root=directory, split='unlabeled', download=True)
STL10(root=directory, split="train", download=True)
STL10(root=directory, split='test', download=True)

