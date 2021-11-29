import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_half_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import PIL
from pdb import set_trace as st
import random
from torchvision.utils import save_image


class HalfDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)
        self.count = 0

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        B_img = Image.open(path).convert('RGB')
        if self.opt.isTrain and not self.opt.no_flip:
            if random.random() > 0.5:
                B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                B_img = B_img
                
        w, h = B_img.size
        rw = random.randint(0, w - self.fineSize)
        rh = random.randint(0, h - self.fineSize)
        # print(rw, rh)
        B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

        w, h = B_img.size
        rw = random.randint(0, int(w/2))
        rh = random.randint(0, int(h/2))

        A_img = B_img.crop((rw, rh, int(rw + w/2), int(rh + h/2)))

        trans = transforms.ToTensor()
        dirname = os.path.dirname(os.path.dirname(path))
        new_dirname = os.path.join(dirname, "taming")
        if not os.path.exists(new_dirname):
            os.mkdir(new_dirname)
        img_name = os.path.join(new_dirname, "B_img" + str(self.count) + ".jpg")
        save_image(trans(B_img), img_name)

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        self.count += 1

        return {'A': A_img, 'B': B_img,
                'A_paths': path, 'B_paths': path,
                'A_start_point':[(rw, rh)]}

    def __len__(self):
        return self.size

    def name(self):
        return 'HalfDataset'
