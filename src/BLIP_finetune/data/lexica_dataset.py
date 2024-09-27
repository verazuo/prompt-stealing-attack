import torch.utils.data as data
import datasets

class LexicaDataset(data.Dataset):
    def __init__(self, dataset_dir, return_text, mode, input_transform=None):
        self.dataset_dir = dataset_dir
        self.data = datasets.load_dataset(dataset_dir, split=mode)
        self.input_transform = input_transform
        print(self.data)

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
        return image, self.return_text_func(index), self.data[index]['modifier10_vector'], self.data[index]['id']

    def getPrompt(self, index):
        return self.data[index]['prompt']

    def getSubject(self, index):
        return self.data[index]['subject']

    def __len__(self):
        return len(self.data)
