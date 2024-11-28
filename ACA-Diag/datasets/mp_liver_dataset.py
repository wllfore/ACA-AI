import torch
import numpy as np
import pandas as pd
from functools import partial
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler
try:
    from datasets.transforms import *
except:
    from transforms import *

class MultiPhaseLiverDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.size = args.img_size
        self.is_training = is_training
        img_list = []
        lab_list = []

        phase_list = ['sweep', 'portal', 'arterial', 'tumor_mask']

        if is_training:
            anno = np.loadtxt(args.train_anno_file, dtype=np.str_)
        else:
            anno = np.loadtxt(args.val_anno_file, dtype=np.str_)

        for item in anno:
            mp_img_list = []
            for phase in phase_list:
                mp_img_list.append(f'{args.data_dir}/{item[0]}/{phase}.nii.gz')
            img_list.append(mp_img_list)
            lab_list.append(item[1])
            
        ### load radiomics
        radiomics_pd = pd.read_csv(args.radiomics_fea_file)
        print (radiomics_pd.shape)

        self.anno = anno
        self.img_list = img_list
        self.lab_list = lab_list
        self.radiomics_pd = radiomics_pd

    def __getitem__(self, index):
        args = self.args
        image = self.load_mp_images(self.img_list[index])
        if self.is_training:
            image = self.transforms(image, args.train_transform_list)
        else:
            image = self.transforms(image, args.val_transform_list)
        image = image.copy()
        label = int(self.lab_list[index])
        
        ### get radiomics feature
        case_id = self.anno[index][0]
        
        try:
            radiomics_idx = self.radiomics_pd[self.radiomics_pd['ID']==case_id].index.item()
            # print (radiomics_idx)
            radio_fea = self.radiomics_pd[:][radiomics_idx:radiomics_idx+1]
            radio_fea.drop(columns=radio_fea.columns[:1], inplace=True)
            radio_fea = radio_fea.to_numpy()
        except ValueError:
            print ('error ', case_id)
        # print (radio_fea)
        
        return (image, label, radio_fea)

    def load_mp_images(self, mp_img_list):
        mp_image = []
        for img in mp_img_list:
            image = load_nii_file(img)
            image = resize3D(image, self.size)
            image = image_normalization(image, win=[-100, 300])     ### normalize set win
            mp_image.append(image[None, ...])
        mp_image = np.concatenate(mp_image, axis=0)
        return mp_image

    def transforms(self, mp_image, transform_list):
        args = self.args
        if 'center_crop' in transform_list:
            mp_image = center_crop(mp_image, args.crop_size)
        if 'random_crop' in transform_list:
            mp_image = random_crop(mp_image, args.crop_size)
        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)
        if 'image_net_autoaugment' in transform_list:
           mp_image = image_net_autoaugment(mp_image)
        return mp_image

    def __len__(self):
        return len(self.img_list)

def create_loader(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
):

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader

if __name__ == "__main__":
    import yaml
    import parser
    import argparse
    from tqdm import tqdm

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        '--data_dir', default='/ailab/user/wanglilong/data/rj_adrenal/All/tumor_image_bbox_new/', type=str)
    parser.add_argument(
        '--train_anno_file', default='/ailab/user/wanglilong/data/rj_adrenal/All/5fold/train_0.txt', type=str)
    parser.add_argument(
        '--val_anno_file', default='/ailab/user/wanglilong/data/rj_adrenal/All/5fold/valid_0.txt', type=str)
    parser.add_argument(
        '--radiomics_fea_file', default='/ailab/user/wanglilong/data/rj_adrenal/All/total_radiomics_norm.csv', type=str)
    
    parser.add_argument('--train_transform_list', default=['random_crop',
                                                           'z_flip',
                                                           'x_flip',
                                                           'y_flip',
                                                           'rotation',],
                        nargs='+', type=str)
    parser.add_argument('--val_transform_list',
                        default=['center_crop'], nargs='+', type=str)
    parser.add_argument('--img_size', default=(16, 128, 128),
                        type=int, nargs='+', help='input image size.')
    parser.add_argument('--crop_size', default=(14, 112, 112),
                        type=int, nargs='+', help='cropped image size.')
    parser.add_argument('--flip_prob', default=0.5, type=float,
                        help='Random flip prob (default: 0.5)')
    parser.add_argument('--angle', default=45, type=int)

    def _parse_args():
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)
        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text
    
    args, args_text = _parse_args()
    args_text = yaml.load(args_text, Loader=yaml.FullLoader)
    args_text['img_size'] = 'xxx'
    print(args_text)

    args.distributed = False
    args.batch_size = 4

    dataset = MultiPhaseLiverDataset(args, is_training=True)
    data_loader = create_loader(dataset, batch_size=4, is_training=True)
    for images, labels, radio_feas in data_loader:
        print(images.shape)
        print(labels)
        print(radio_feas.shape)

    # val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    # val_data_loader = create_loader(val_dataset, batch_size=10, is_training=False)
    # for images, labels in val_data_loader:
    #     print(images.shape)
    #     print(labels)