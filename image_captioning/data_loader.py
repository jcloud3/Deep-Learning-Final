from nltk import tokenize
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image

from .preprocessing.build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """
    COCO Custom Dataset compatible with torch.utils.data.DataLoader.

    Constructor Params
    :param: root            - image dir
    :param: json            - coco annotation json path
    :param: vocab           - Vocabulary Object
    :kwarg: transform       - image transformer
    :kwarg: validation_mode - decides if you need caption text or tokens
    """

    def __init__(self, root, json, vocab, transform=None, validation_mode=False):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.validation_mode = validation_mode

    def __getitem__(self, index):
        """
        Returns one data pair (image and caption)
        If validation mode, the caption will not be tokenized

        Get single Image Entry
            - Image
            - Tokenized Caption
            - Text Caption (for calculating bleu)
        """
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]

        # Get Image and Caption Text
        caption_text = coco.anns[ann_id]["caption"]
        img_id = coco.anns[ann_id]["image_id"]
        path = coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.validation_mode:
            return image, caption_text

        # Convert Caption Text to Tokenized Captions
        tokens = nltk.tokenize.word_tokenize(str(caption_text).lower())
        caption_tokens = []
        caption_tokens.append(vocab("<start>"))
        caption_tokens.extend([vocab(token) for token in tokens])
        caption_tokens.append(vocab("<end>"))
        target = torch.Tensor(caption_tokens)

        return image, target

    def __len__(self):
        return len(self.ids)


def get_collate_fn(validation_mode):
    def collate_fn(data):
        """Creates mini-batch tensors from the list of tuples (image, caption).

        We should build custom collate_fn rather than using default collate_fn,
        because merging caption (including padding) is not supported in default.

        Args:
            data: list of tuple (image, caption_tokens, caption_text).
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)

        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        if validation_mode:
            return images, captions

        caption_tokens = captions

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in caption_tokens]

        target_tokens = torch.zeros(len(caption_tokens), max(lengths)).long()
        # target_caption_text = []
        for i, cap in enumerate(caption_tokens):
            end = lengths[i]
            target_tokens[i, :end] = cap[:end]
            # target_caption_text.append()

        return images, target_tokens, lengths

    return collate_fn


def get_loader(
    root,
    json,
    vocab,
    transform,
    batch_size,
    shuffle,
    num_workers,
    validation_mode=False,
):
    """
    Returns torch.utils.data.DataLoader for custom coco dataset.

    If validation mode is True, the Data Loader will not return tokenized
    captions and will not return the length
    """
    # COCO caption dataset
    coco = CocoDataset(
        root=root,
        json=json,
        vocab=vocab,
        transform=transform,
        validation_mode=validation_mode,
    )

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(
        dataset=coco,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=get_collate_fn(validation_mode),
    )
    return data_loader
