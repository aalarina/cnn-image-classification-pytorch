from torchvision import transforms

def get_transforms(img_size=224):
    """
    Returns training and validation transforms.
    
    - transform_0: augmentation for class 0 (artifact images)
    - transform_1: minimal transform for class 1 (clean images)
    - val_transform: used for validation and test (no augmentation)
    """

    # More aggressive augmentation for class 0 (to handle imbalance)
    transform_0 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Minimal transformation for class 1 (clean images)
    transform_1 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation / test transform (no augmentation!)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform_0, transform_1, val_transform
