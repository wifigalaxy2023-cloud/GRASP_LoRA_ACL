from dataclasses import dataclass

@dataclass
class custom_whole_dataset:
    dataset: str = "custom_whole_dataset"
    train_split: str = "train"
    test_split: str = "val"
    dataset_name: str = "english_qa"


@dataclass
class custom_few_dataset:
    dataset: str = "custom_few_dataset"
    train_split: str = "train"
    test_split: str = "val"
    dataset_name: str = "english_qa"
    example_num: int = 50
