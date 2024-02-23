import os
from pathlib import Path
from typing import Union, Callable, Optional
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
import random

from data.utils import (
    tokenize_prompt,
    ToRGB,
    ColorBackgroundRGBA,
    CropForegroundRGBA,
    RandomResizeForeground,
)


class GenDataset(Dataset):
    def __init__(
        self,
        data_root: Union[str, os.PathLike],
        tokenizer: Callable,
        image_norm_transform: Optional[Callable] = None,
        tokenizer_max_length: Optional[int] = None,
        refimage_norm_transform: Optional[Callable] = None,
        resolution: int = 512,
        empty_prompt_p=0.0,
        tokenizer_2=None,
        jitter_p=0.1,
        erase_p=0.4,
        min_pct_area=0.4,
    ):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data {self.data_root} images root doesn't exists.")

        objects = os.listdir(self.data_root)
        self.objects = objects
        self.num_images = len(self.objects)

        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_2 = tokenizer_2
        self.empty_prompt_p = empty_prompt_p
        self.resolution = resolution

        self.jitter_transform = transforms.RandomApply(
            torch.nn.ModuleList([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)]),
            p=jitter_p,
        )
        self.erase = transforms.RandomErasing(p=erase_p, scale=(0.02, 0.25), value=1)

        self.image_transform = transforms.Resize(resolution)
        self.ref_image_transform = transforms.Compose(
            [
                ColorBackgroundRGBA(),
                CropForegroundRGBA(),
                ToRGB(),
                RandomResizeForeground(resolution, min_pct_area=min_pct_area),
                transforms.RandomHorizontalFlip(),
            ]
        )

        self.image_norm_transform = image_norm_transform
        self.refimage_norm_transform = refimage_norm_transform

    @property
    def _length(self):
        return len(self.objects)

    def __len__(self):
        return self._length

    def _prepare_image(self, image_path, transform):
        if isinstance(image_path, Image.Image):
            return transform(image)
        elif isinstance(image_path, np.ndarray):
            return transform(Image.fromarray(image_path, mode="RGBA"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            image = Image.open(image_path)
            image = exif_transpose(image)
            if np.std(np.array(image)) == 0:
                raise ValueError
        return transform(image)

    def __getitem__(self, index):
        try:
            example = {}
            object = self.objects[index]
            images = [
                img
                for img in list(Path(self.data_root, object).iterdir())
                if "_generation.png" in str(img)
            ]

            target = random.sample(images, 1)[0]
            prompt_fname = Path(str(target).replace("_generation.png", "_caption.txt"))
            with open(prompt_fname, "r") as f:
                prompt = f.read()

            refprompt = prompt.split(",")[0]

            if np.random.rand() < self.empty_prompt_p:
                prompt = ""

            text_inputs = tokenize_prompt(
                self.tokenizer, prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            reftext_inputs = tokenize_prompt(
                self.tokenizer,
                refprompt,
                tokenizer_max_length=self.tokenizer_max_length,
            )
            example["prompt"] = prompt
            example["refprompt"] = refprompt
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_refprompt_ids"] = reftext_inputs.input_ids
            if self.tokenizer_2 is not None:
                text_inputs_2 = tokenize_prompt(self.tokenizer_2, prompt)
                example["instance_prompt_ids2"] = text_inputs_2.input_ids
                reftext_inputs_2 = tokenize_prompt(self.tokenizer_2, refprompt)
                example["instance_refprompt_ids2"] = reftext_inputs_2.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
            example["path"] = target
            example["original_instance_images"] = self._prepare_image(
                target, self.image_transform
            )
            example["instance_images"] = self.image_norm_transform(
                example["original_instance_images"]
            )
            example["original_reference_images"] = []
            example["reference_images"] = []
            example["reference_paths"] = [target]

            refpath = target
            im = np.array(Image.open(refpath))
            mask = np.array(
                Image.open(str(refpath).replace("generation.png", "mask.png"))
            )
            im = np.concatenate((im, mask[:, :, np.newaxis]), axis=-1)
            example["original_reference_images"].append(
                self._prepare_image(im, self.ref_image_transform)
            )
            example["reference_images"] = [
                self.erase(self.refimage_norm_transform(self.jitter_transform(im)))
                for im in example["original_reference_images"]
            ]
            example["original_sizes"] = self.resolution
        except Exception as e:
            print("Data Error:")
            print(e)
            return self.__getitem__(np.random.randint(self._length))
        return example
