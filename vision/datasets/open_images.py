import numpy as np
import pathlib
import cv2
import pandas as pd


class OpenImagesDataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", data_filter=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self.data_filter = data_filter

        self.data, self.class_names, self.class_dict = self._read_data()

    def __getitem__(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        boxes = image_info['boxes']
        labels = image_info['labels']
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def _read_data(self):
        annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations-bbox.csv"
        annotations = pd.read_csv(annotation_file)
        class_names = list(annotations['ClassName'].unique())
        class_dict = {class_name: i + 1 for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            labels = np.array([class_dict[name] for name in group["ClassName"]])
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
