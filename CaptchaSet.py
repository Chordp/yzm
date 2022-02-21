import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import Utils
import string
import Utils
from captcha.image import ImageCaptcha
import random
characters = '-' + string.digits + string.ascii_lowercase
class CaptchaSet(Dataset):
    def __init__(self,root_dir):
        super().__init__()
        self.captcha_path = [os.path.join(root_dir,x) for x in os.listdir(root_dir)]
        print(len(self.captcha_path))
        self.transforms = transforms.Compose([
            transforms.Resize((80,160)),
            transforms.ToTensor(),
            # transforms.Grayscale()
            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    def __len__(self):
        return len(self.captcha_path)
    def __getitem__(self,idx):
        path = self.captcha_path[idx]
        image = self.transforms(Image.open(fp=path))
        label = path.split('\\')[-1].split('_')[0]
        target = torch.tensor([characters.find(x) for x in label], dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=5, dtype=torch.long)
        input_lengths = torch.full(size=(1, ), fill_value=15, dtype=torch.long)
        return image, target,input_lengths ,target_length
class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        image = transforms.functional.to_tensor(self.generator.generate_image(random_str))
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length

def test():
    from torchvision.transforms.functional import  to_pil_image
    train_set = CaptchaSet(root_dir='./data/train')
    a,b = train_set[0]
    aaa = Utils.strLabelConverter(characters)
    print(aaa.encode('123456')[1])
# test()