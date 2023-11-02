import os

import cv2
import torch.utils.data as data


class KideneyContentDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.content_root = os.path.join(root, "A", "train")
        else:
            self.content_root = os.path.join(root, "A", "test")
        assert os.path.exists(self.content_root), f"path '{self.content_root}' does not exist."

        content_names = [p for p in os.listdir(self.content_root) if p.endswith(".png")]
        assert len(content_names) > 0, f"not find any contents in {self.content_root}."

        self.contents_path = [os.path.join(self.content_root, n) for n in content_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        content_path = self.contents_path[idx]
        content = cv2.imread(content_path, flags=cv2.IMREAD_COLOR)
        assert content is not None, f"failed to read content: {content_path}"
        content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = content.shape

        if self.transforms is not None:
            content = self.transforms(content)

        return content

    def __len__(self):
        return len(self.contents_path)


class KidneyStyleDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.style_root = os.path.join(root, "B", "train")
        else:
            self.style_root = os.path.join(root, "B", "test")
        assert os.path.exists(self.style_root), f"path '{self.style_root}' does not exist."

        style_names = [p for p in os.listdir(self.style_root) if p.endswith(".png")]
        assert len(style_names) > 0, f"not find any contents in {self.content_root}."

        self.styles_path = [os.path.join(self.style_root, n) for n in style_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        style_path = self.styles_path[idx]
        style = cv2.imread(style_path, flags=cv2.IMREAD_COLOR)
        assert style is not None, f"failed to read style: {style_path}"
        style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = style.shape

        if self.transforms is not None:
            style = self.transforms(style)

        return style

    def __len__(self):
        return len(self.styles_path)


class KidneyDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.content_root = os.path.join(root, "A", "train")
            self.style_root = os.path.join(root, "B", "train")
        else:
            self.content_root = os.path.join(root, "A", "test")
            self.style_root = os.path.join(root, "B", "test")
        assert os.path.exists(self.content_root), f"path '{self.content_root}' does not exist."
        assert os.path.exists(self.style_root), f"path '{self.style_root}' does not exist."

        content_names = [p for p in os.listdir(self.content_root) if p.endswith(".png")]
        style_names = [p for p in os.listdir(self.style_root) if p.endswith(".png")]
        assert len(content_names) > 0, f"not find any contents in {self.content_root}."

        # check contents and style
        re_style_names = []
        for p in content_names:
            style_name = p
            assert style_name in style_names, f"{p} has no corresponding style."
            re_style_names.append(style_name)
        style_names = re_style_names

        self.contents_path = [os.path.join(self.content_root, n) for n in content_names]
        self.styles_path = [os.path.join(self.style_root, n) for n in style_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        content_path = self.contents_path[idx]
        style_path = self.styles_path[idx]
        content = cv2.imread(content_path, flags=cv2.IMREAD_COLOR)
        assert content is not None, f"failed to read content: {content_path}"
        content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = content.shape

        style = cv2.imread(style_path, flags=cv2.IMREAD_COLOR)
        assert style is not None, f"failed to read style: {style_path}"
        style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        if self.transforms is not None:
            content, style = self.transforms(content, style)

        return content, style

    def __len__(self):
        return len(self.contents_path)

    @staticmethod
    def collate_fn(batch):
        contents, styles = list(zip(*batch))
        batched_imgs = cat_list(contents, fill_value=0)
        batched_styles = cat_list(styles, fill_value=0)

        return batched_imgs, batched_styles


def cat_list(contents, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in contents]))
    batch_shape = (len(contents),) + max_size
    batched_imgs = contents[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(contents, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    
    train_dataset = KidneyDataset("D:/zy/project/virtual_staining/data/kidney_trans/", train=True)
    print(len(train_dataset))
    
    content_dataset = KideneyContentDataset("D:/zy/project/virtual_staining/data/kidney_trans/", train=True)
    print(len(content_dataset))

    # val_dataset = DUTSDataset("./", train=False)
    # print(len(val_dataset))

    content = content_dataset[0]
    print(content.shape)
    print(content)
    
    style_dataset = KidneyStyleDataset("D:/zy/project/virtual_staining/data/kidney_trans/", train=True)
    print(len(style_dataset))

    # val_dataset = DUTSDataset("./", train=False)
    # print(len(val_dataset))

    style= style_dataset[0]
    print(style.shape)
    print(style)
