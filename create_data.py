import random
from pathlib import Path
import shutil


def get_subset(data_path,
               data_splits=["train", "test", "val"],
               classes=["NORMAL", "PNEUMONIA"],
               amount=0.1,
               seed=42):
    random.seed(42)
    label_splits = {}

    for data_split in data_splits:
        label_path = data_path / data_split
        classes_dict = {}
        for class_name in classes:
            img_paths = label_path / class_name
            img_list = list(Path(img_paths).glob("*.jpeg"))
            size = int(amount*len(img_list))
            sampled_images = random.sample(population=img_list, k=size)
            classes_dict[class_name] = sampled_images
        label_splits[data_split] = classes_dict

    return label_splits


def copy(paths,
         target_dir):
    for split_name in paths:
        for class_name in paths[split_name]:
            dest_dir = target_dir / split_name / class_name
            if not dest_dir.is_dir():
                dest_dir.mkdir(parents=True, exist_ok=True)
            for img in paths[split_name][class_name]:
                str_img = str(img)
                path_parts = str_img.split("\\")
                dest_img_path = dest_dir / path_parts[-1]
                shutil.copy2(src=img, dst=dest_img_path)


def main():

    data_path = Path("data")
    full_data = data_path / "full_data"
    subset_size = 0.1
    label_path = get_subset(data_path=full_data,
                            amount=subset_size)

    taget_dir = data_path / "10_percent_data"

    copy(paths=label_path,
         target_dir=taget_dir)


if __name__ == "__main__":
    main()
