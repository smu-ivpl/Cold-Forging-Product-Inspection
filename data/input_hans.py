import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config
import glob

def read_split(num_segmented: int, kind: str):
    fn = f"KSDD2/split_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples
        elif kind == 'TEST':
            return test_samples
        else:
            raise Exception('Unknown')


class HSDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(HSDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

        # img_path = '/home/yjkim/HANSUNG/datasets/stabbed_data_1013/*/*'

        # img_path = f'/home/yjkim/HANSUNG/datasets/hs_data/{self.kind.lower()}/'
        img_path = os.path.join(self.cfg.DATASET_PATH, self.kind.lower())
        # img_path = f'/home/yjkim/HANSUNG/datasets/hs_data/shape1/{self.kind.lower()}/'
        file_list = os.listdir(img_path)
        # dirs = glob.glob(img_path)
        data_points = []

        paths = [file for file in file_list if file.endswith('Normal.png')]
        # paths = [file for file in file_list if file.endswith('Shape1.png')]

        '''
        seg = False
        if self.kind.lower() == 'test':
            seg = True
        '''
        for num, path in enumerate(paths):
            data_points.append((path.split('/')[-1][:-4], True))  # seg))
            # imgname = path.split('/')[-1][:-10] + 'Shape1'
            # data_points.append((path.split('/')[-1][:-14], True))  # seg))
            # data_points.append((imgname, True))  # seg))

        # data_points = read_split(self.cfg.NUM_SEGMENTED, self.kind)
        print(f'{self.kind}_data_point: ', data_points)

        for part, is_segmented in data_points:
            image_path = os.path.join(self.path, self.kind.lower(), f"{part}.png")
            rf_path = os.path.join('/home/yjkim/HANSUNG/datasets/hs_data/SpecularRF_new/crop_total/', f"{part[:-6]}RF.png")
            seg_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_mask.png")
            # seg_mask_path = os.path.join(self.path, self.kind.lower(), f"{part[:-7]}_Normal_mask.png")

            import cv2
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            rf_image = self.read_img_resize(rf_path, self.grayscale, self.image_size)
            image = cv2.hconcat([image, rf_image])
            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            if positive:
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
            else:
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()




'''
def read_split(train_num: int, num_segmented: int, fold: int, kind: str):
    fn = f"KSDD/split_{train_num}_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples[fold]
        elif kind == 'TEST':
            return test_samples[fold]
        else:
            raise Exception('Unknown')
'''
'''
import numpy as np
import pickle
import os
from .dataset import Dataset
from config import Config


class HSDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(HSDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []
        is_segmented = True

        # dirs = sorted(os.listdir(os.path.join(self.path)))
        # images = [file for file in dirs if file.endswith('.jpg')]

        for sample in sorted(os.listdir(os.path.join(self.path))):  # images:
            if not sample.__contains__('mask'):
                image_path = self.path + sample
                seg_mask_path = f"{image_path[:-4]}_mask.png"
                image = self.read_img_resize(image_path, self.grayscale, self.image_size)
                seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, dilate=self.cfg.DILATE)
                sample_name = f"{sample}"[:-4]
                if positive:
                    image = self.to_tensor(image)
                    seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                    seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                    seg_mask = self.to_tensor(self.downsize(seg_mask))
                    pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, sample_name))
                else:
                    image = self.to_tensor(image)
                    seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                    seg_mask = self.to_tensor(self.downsize(seg_mask))
                    neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, sample_name))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()
'''