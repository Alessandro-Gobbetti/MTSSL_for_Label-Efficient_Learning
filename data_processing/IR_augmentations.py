import torch
import torchvision.transforms as transforms


class TransformationsContrastiveRotation(object):

    def __init__(self):
        self.transform_start = transforms.Compose(
            [transforms.RandomHorizontalFlip()])
        # apply random patch of the image

        self.rotation_90 = transforms.Compose(
            [transforms.RandomRotation(degrees=(90, 90))])
        self.rotation_180 = transforms.Compose(
            [transforms.RandomRotation(degrees=(180, 180))])
        self.rotation_270 = transforms.Compose(
            [transforms.RandomRotation(degrees=(270, 270))])

        self.transform_end = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))])

        self.contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                       transforms.RandomApply([
                                                           transforms.RandomRotation(
                                                               degrees=(90, 270))
                                                       ], p=0.2),
                                                       transforms.RandomApply([
                                                           transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.RandomErasing(p=1.0, scale=(
                                                                   0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
                                                               transforms.ToPILImage()
                                                           ])], p=0.2),
                                                       transforms.RandomResizedCrop(
                                                           size=96),
                                                       transforms.RandomApply([
                                                           transforms.ColorJitter(
                                                               brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                                                       ], p=0.8),
                                                       transforms.RandomGrayscale(
                                                           p=0.2),
                                                       transforms.RandomApply([
                                                           transforms.GaussianBlur(
                                                               kernel_size=5, sigma=(0.1, 2.0))
                                                       ], p=0.2),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(
                                                           (0.5,), (0.5,))
                                                       ])

    def __call__(self, x):
        # returns the transformed image 3 boolean values indicating whether the image is transformed with color, gray, or blur
        x1 = self.contrast_transforms(x)
        x2 = self.contrast_transforms(x)

        # apply the initial transformation
        x = self.transform_start(x)
        # create a transformation that either rotates the image by 90, 180, or 270 degrees
        # pick an int in [0, 1, 2, 3]
        rotation_choice = torch.randint(0, 4, (1,)).item()
        if rotation_choice == 1:
            # rotate by 90 degrees
            x = self.rotation_90(x)
        elif rotation_choice == 2:
            # rotate by 180 degrees
            x = self.rotation_180(x)
        elif rotation_choice == 3:
            # rotate by 270 degrees
            x = self.rotation_270(x)

        x = self.transform_end(x)
        return x1, x2, x, float(rotation_choice)


class TransformationsContrastiveMasked(object):

    def __init__(self):

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.ColorJitter(brightness=0.5,
                                   contrast=0.5,
                                   saturation=0.5,
                                   hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9)], p=0.5),
            transforms.ToTensor(),
        ])

        self.transform_patch = transforms.Compose([
            transforms.RandomErasing(p=1.0, scale=(
                0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
        ])

        self.transform_end = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                       transforms.RandomApply([
                                                           transforms.RandomRotation(
                                                               degrees=(90, 270))
                                                       ], p=0.2),
                                                       transforms.RandomApply([
                                                           transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.RandomErasing(p=1.0, scale=(
                                                                   0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
                                                               transforms.ToPILImage()
                                                           ])], p=0.2),
                                                       transforms.RandomResizedCrop(
                                                           size=96),
                                                       transforms.RandomApply([
                                                           transforms.ColorJitter(
                                                               brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                                                       ], p=0.8),
                                                       transforms.RandomGrayscale(
                                                           p=0.2),
                                                       transforms.RandomApply([
                                                           transforms.GaussianBlur(
                                                               kernel_size=9, sigma=(0.1, 2.0))
                                                       ], p=0.2),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(
                                                           (0.5,), (0.5,))
                                                       ])

    def __call__(self, x):
        # returns the transformed image 3 boolean values indicating whether the image is transformed with color, gray, or blur
        x1 = self.contrast_transforms(x)
        x2 = self.contrast_transforms(x)

        x = self.transform(x)
        x_masked = self.transform_patch(x)
        x_masked = self.transform_end(x_masked)
        x_original = self.transform_end(x)

        return x1, x2, x_masked, x_original


class TransformationsMasked(object):

    def __init__(self):
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.ColorJitter(brightness=0.5,
                                   contrast=0.5,
                                   saturation=0.5,
                                   hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
            transforms.ToTensor(),
        ])

        self.transform_patch = transforms.Compose([
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
        ])

        self.transform_end = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
        ])
        


    def __call__(self, x):
        x = self.transform(x)
        x_masked = self.transform_patch(x)
        x_masked = self.transform_end(x_masked)
        x_original = self.transform_end(x)

        return x_masked, x_original
    
class TransformationsRotation(object):

    def __init__(self):
        

        self.transform_start = transforms.Compose([transforms.RandomHorizontalFlip()])
    
        self.rotation_90 = transforms.Compose([transforms.RandomRotation(degrees=(90, 90))])
        self.rotation_180 = transforms.Compose([transforms.RandomRotation(degrees=(180, 180))])
        self.rotation_270 = transforms.Compose([transforms.RandomRotation(degrees=(270, 270))])

        self.transform_end = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))])



    def __call__(self, x):
        
        # apply the initial transformation
        x = self.transform_start(x)
        # create a transformation that either rotates the image by 90, 180, or 270 degrees
        # pick an int in [0, 1, 2, 3]
        rotation_choice = torch.randint(0, 4, (1,)).item()
        if rotation_choice == 1:
            # rotate by 90 degrees
            x = self.rotation_90(x)
        elif rotation_choice == 2:
            # rotate by 180 degrees
            x = self.rotation_180(x)
        elif rotation_choice == 3:
            # rotate by 270 degrees
            x = self.rotation_270(x)

        x = self.transform_end(x)
        return  x, float(rotation_choice)
    
















class UnifiedTransform:
    """
    Single transformation chooser.

    mode (str): one of
      - "contrastive_rotation" : returns (x1, x2, x, rotation_choice)
      - "contrastive_masked"   : returns (x1, x2, x_masked, x_original)
      - "masked"               : returns (x_masked, x_original)
      - "rotation"             : returns (x, rotation_choice)
      - "contrastive"          : returns (x1, x2)  (simple contrastive pair)

    If both mode and boolean flags (contrastive, rotation, masked) are passed,
    mode takes precedence. Input expected as PIL image.
    """
    def __init__(self, mode: str = "contrastive_rotation",
                 contrastive: bool = False, rotation: bool = False, masked: bool = False):
        if mode not in {"contrastive_rotation", "contrastive_masked", "masked", "rotation", "contrastive"}:
            # fallback to booleans if mode invalid
            if contrastive:
                mode = "contrastive"
            elif rotation and contrastive:
                mode = "contrastive_rotation"
            elif masked and contrastive:
                mode = "contrastive_masked"
            elif masked:
                mode = "masked"
            elif rotation:
                mode = "rotation"
            else:
                mode = "contrastive_rotation"
        self.mode = mode

        # basic building blocks (mirrors existing classes)
        self.transform_start = transforms.Compose([transforms.RandomHorizontalFlip()])

        self.rotation_90 = transforms.Compose([transforms.RandomRotation(degrees=(90, 90))])
        self.rotation_180 = transforms.Compose([transforms.RandomRotation(degrees=(180, 180))])
        self.rotation_270 = transforms.Compose([transforms.RandomRotation(degrees=(270, 270))])

        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
            transforms.ToTensor(),
        ])

        self.transform_patch = transforms.Compose([
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
        ])

        # contrast transforms used by contrastive variants
        self.contrast_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(degrees=(90, 270))], p=0.2),
            transforms.RandomApply([
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
                    transforms.ToPILImage()
                ])], p=0.2),
            transforms.RandomResizedCrop(size=96),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # keep the contrastive simple pair mode option
        self.simple_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _random_rotation_apply(self, img):
        rotation_choice = torch.randint(0, 4, (1,)).item()
        if rotation_choice == 1:
            img = self.rotation_90(img)
        elif rotation_choice == 2:
            img = self.rotation_180(img)
        elif rotation_choice == 3:
            img = self.rotation_270(img)
        img = self.normalize(img)
        return img, float(rotation_choice)

    def __call__(self, x):
        if self.mode == "contrastive_rotation":
            x1 = self.contrast_transforms(x)
            x2 = self.contrast_transforms(x)
            # rotation applied to 'x' (the augmented view used for rotation task)
            x_rot, rot_choice = self._random_rotation_apply(self.transform_start(x))
            return x1, x2, x_rot, rot_choice

        if self.mode == "contrastive_masked":
            x1 = self.contrast_transforms(x)
            x2 = self.contrast_transforms(x)

            x_trans = self.transform(x)
            x_masked = self.transform_patch(x_trans)
            x_masked = self.normalize(x_masked)
            x_original = self.normalize(x_trans)
            return x1, x2, x_masked, x_original

        if self.mode == "masked":
            x_trans = self.transform(x)
            x_masked = self.transform_patch(x_trans)
            x_masked = self.normalize(x_masked)
            x_original = self.normalize(x_trans)
            return x_masked, x_original

        if self.mode == "rotation":
            x_rot, rot_choice = self._random_rotation_apply(self.transform_start(x))
            return x_rot, rot_choice

        # default: simple contrastive pair
        x1 = self.simple_augmentation(x)
        x2 = self.simple_augmentation(x)
        return x1, x2
    



class UnifiedTransform:
    """
    Single transformation chooser.

    Always returns a 6-tuple:
      (x1, x2, x_rot, rot_label, x_masked, x_original)
    Depending on the mode, some of these may be None.
    """
    def __init__(self,
                 is_contrastive: bool = False, is_rotation: bool = False, is_mask: bool = False):
        
        assert sum([is_contrastive, is_rotation, is_mask]) > 0, "At least one transformation type must be enabled."
        
        self.is_contrastive = is_contrastive
        self.is_rotation = is_rotation
        self.is_mask = is_mask

        # building blocks
        self.transform_start = transforms.Compose([transforms.RandomHorizontalFlip()])

        self.rotation_90 = transforms.Compose([transforms.RandomRotation(degrees=(90, 90))])
        self.rotation_180 = transforms.Compose([transforms.RandomRotation(degrees=(180, 180))])
        self.rotation_270 = transforms.Compose([transforms.RandomRotation(degrees=(270, 270))])

        # Normalizers: one for PIL->tensor+norm, one for already-tensor normalization
        self.normalize_pil = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))])
        self.normalize_tensor = transforms.Normalize((0.5,), (0.5,))

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
            transforms.ToTensor(),
        ])

        self.transform_patch = transforms.Compose([
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
        ])

        # contrast transforms already yield normalized tensors
        self.contrast_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(degrees=(90, 270))], p=0.2),
            transforms.RandomApply([
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),
                    transforms.ToPILImage()
                ])], p=0.2),
            transforms.RandomResizedCrop(size=96),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.simple_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _random_rotation_apply(self, img):
        """
        img: PIL image (not tensor). Returns (tensor_normalized, int_label)
        """
        rotation_choice = torch.randint(0, 4, (1,)).item()
        if rotation_choice == 1:
            img = self.rotation_90(img)
        elif rotation_choice == 2:
            img = self.rotation_180(img)
        elif rotation_choice == 3:
            img = self.rotation_270(img)
        # normalize PIL -> tensor
        return self.normalize_pil(img), float(rotation_choice)

    def __call__(self, x):
        dummy_tensor = torch.tensor(0.0)

        # defaults
        x1, x2 = dummy_tensor, dummy_tensor
        x_rot, rot_label = dummy_tensor, dummy_tensor
        x_mask, x_original = dummy_tensor, dummy_tensor

        if self.is_contrastive:
            x1 = self.contrast_transforms(x)
            x2 = self.contrast_transforms(x)

        if self.is_rotation:
            x_rot, rot_label = self._random_rotation_apply(self.transform_start(x))

        if self.is_mask:
            x_trans = self.transform(x)
            x_mask = self.transform_patch(x_trans)
            x_mask = self.normalize_tensor(x_mask)
            x_original = self.normalize_tensor(x_trans)

        # return fixed-length tuple with unused slots as dummy tensors
        return x1, x2, x_rot, rot_label, x_mask, x_original