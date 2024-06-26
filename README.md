# Prompt Stealing Attacks Against Text-to-Image Generation Models

[![hugging](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/vera365/lexica_dataset)
[![arXiv: paper](https://img.shields.io/badge/arXiv-paper-red.svg)](https://arxiv.org/abs/2302.09923)
[![license](https://img.shields.io/badge/License-CC_BY_4.0/MIT-blue)](#license)

This is the official implementation of the USENIX 2024 paper [Prompt Stealing Attacks Against Text-to-Image Generation Models](https://arxiv.org/abs/2302.09923).

## LexicaDataset

LexicaDataset is a large-scale text-to-image prompt dataset containing **61,467 prompt-image pairs** collected from [Lexica](https://lexica.art/). All prompts are curated by real users and images are generated by Stable Diffusion.

LexicaDataset is available at [🤗 Hugging Face Datasets](https://huggingface.co/datasets/vera365/lexica_dataset).

**Load LexicaDataset**

You can use the Hugging Face [`Datasets`](https://huggingface.co/docs/datasets/quickstart) library to easily load prompts and images from LexicaDataset.

```python
import numpy as np
from datasets import load_dataset

trainset = load_dataset('vera365/lexica_dataset', split='train')
testset  = load_dataset('vera365/lexica_dataset', split='test')
```

**Metadata Schema**

`trainset` and `testset` share the same schema.

| Column              | Type       | Description                                                  |
| :------------------ | :--------- | :----------------------------------------------------------- |
| `image`             | `image`    | The generated image                                          |
| `prompt`            | `string`   | The text prompt used to generate this image                  |
| `id`                | `string`   | Image UUID                                                   |
| `promptid`          | `string`   | Prompt UUID                                                  |
| `width`             | `uint16`   | Image width                                                  |
| `height`            | `uint16`   | Image height                                                 |
| `seed`              | `uint32`   | Random seed used to generate this image.                     |
| `grid`              | `bool`     | Whether the image is composed of multiple smaller images arranged in a grid |
| `model`             | `string`   | Model used to generate the image                             |
| `nsfw`              | `string`   | Whether the image is NSFW                                    |
| `subject`           | `string`   | the subject/object depicted in the image, extracted from the prompt |
| `modifier10`        | `sequence` | Modifiers in the prompt that appear more than 10 times in the whole dataset. We regard them as labels to train the modifier detector |
| `modifier10_vector` | `sequence` | One-hot vector of `modifier10`                               |


## Code

Code will be released soon!

## Ethics & Disclosure

According to the [terms and conditions of Lexica](https://lexica.art/terms), images on the website are available under the Creative Commons Noncommercial 4.0 Attribution International License. We strictly followed Lexica’s Terms and Conditions, utilized only the official Lexica API for data retrieval, and disclosed our research to Lexica. We also responsibly disclosed our findings to related prompt marketplaces.

## License

LexicaDataset is available under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). The code in this repository is available under the [MIT License](./LICENSE).

**Note, the code is intended for research purposes only. Any misuse is strictly prohibited.**

## Citation

If you find this useful in your research, please consider citing:

```bibtex
@inproceedings{SQBZ24,
  author = {Xinyue Shen and Yiting Qu and Michael Backes and Yang Zhang},
  title = {{Prompt Stealing Attacks Against Text-to-Image Generation Models}},
  booktitle = {{USENIX Security Symposium (USENIX Security)}},
  publisher = {USENIX},
  year = {2024}
}
```

