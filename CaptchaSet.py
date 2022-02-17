import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import Utils
import string
import Utils
characters = '-' + string.digits + string.ascii_lowercase
class CaptchaSet(Dataset):
    def __init__(self,root_dir):
        super().__init__()
        self.captcha_path = [os.path.join(root_dir,x) for x in os.listdir(root_dir)]
        print(len(self.captcha_path))
        self.transforms = transforms.Compose([
            transforms.Resize((60,160)),
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
        input_lengths = torch.full(size=(1, ), fill_value=10, dtype=torch.long)
        return image, target,input_lengths ,target_length

def test():
    from torchvision.transforms.functional import  to_pil_image
    train_set = CaptchaSet(root_dir='./data/train')
    a,b = train_set[0]
    aaa = Utils.strLabelConverter(characters)
    print(aaa.encode('123456')[1])
# test()