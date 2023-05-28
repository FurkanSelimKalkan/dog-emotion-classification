class Config:
    # Dataset settings
    data_dir = "../../data for emotion classification/dataset_cleaned"

    # Model settings
    num_classes = 4

    # Training settings
    batch_size = 32
    num_epochs = 6
    learning_rate = 0.001

    # Device settings
    use_gpu = True  # Set to False if you want to use CPU
    num_cpu_workers = 14  # Number of CPU workers for data loading
    l2_regularization = 0.001  # Add the l2_regularization attribute and set its value
