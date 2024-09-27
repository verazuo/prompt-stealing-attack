import torch.utils.data as data
import datasets
import torch
import pandas as pd 
import os
import numpy as np

class LexicaDataset(data.Dataset):
    def __init__(self, dataset_dir, return_text, mode, input_transform=None):
        self.dataset_dir = dataset_dir
        self.data = datasets.load_dataset(dataset_dir, split=mode, cache_dir="./data/lexica_dataset/")
        self.input_transform = input_transform
        print(self.data)

        keyword_df = pd.read_csv(os.path.join("./data/lexica_keyword_test_10.csv"), header=0)
        keyword_list = keyword_df['keyword'].tolist()
        self.category_map = {keyword_list[i]: i+1 for i in range(len(keyword_list))}
        self.label2category = {v: k for k, v in self.category_map.items()}
	
        print("Return text:", return_text)
        if return_text == 'prompt':
            self.return_text_func = self.getPrompt
        elif return_text == 'subject':
            self.return_text_func = self.getSubject
        else:
            raise NotImplementedError
        
    def __getitem__(self, index):
        image = self.data[index]['image']
        if self.input_transform:
            image = self.input_transform(image)
        modifier10_vector = torch.Tensor(self.data[index]['modifier10_vector'])
        return image, self.return_text_func(index), modifier10_vector, self.data[index]['id']

    def getPrompt(self, index):
        return self.data[index]['prompt']

    def getSubject(self, index):
        return self.data[index]['subject']

    def __len__(self):
        return len(self.data)

    def getCategoryListByArray(self, item):
        # Usage: self.getCategoryListByArray(self.labels[index])
        # Input: [1,0,0,1,...,0], here 1 means the category is present in the image
        # Return: ['dog', 'person', 'bicycle']
        item_categories = set()
        for idx, t in enumerate(item):
            if t == 1:
                item_categories.add(self.label2category[idx+1])
        return list(item_categories)

    def getCategoryListByIndices(self, indices):
        categories = set()
        for idx in indices:
            categories.add(self.label2category[idx+1])
        return list(categories)

    def getLabelVector(self, categories):
        # input: ["person", ...]
        # output: [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        label = np.zeros(len(self.category_map))
        for c in categories:
            index = self.category_map[str(c)]-1
            label[index] = 1.0 # / label_num
        return label