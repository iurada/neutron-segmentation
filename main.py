import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import os
import logging
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
from parse_args import parse_arguments

from datasets import Cityscapes, GTA5
from datasets.utils import SeededDataLoader
import utils.ext_transform as et
from models.deeplabv3_resnet import deeplabv3_resnet50
from lib.hg_injector import Injector
from metrics.stream_metrics import StreamSegMetrics

from globals import CONFIG

@torch.no_grad()
def evaluate(model: nn.Module, injector: Injector, data: SeededDataLoader, inject: bool=False):
    model.eval()
    injector.inject = inject
    
    meter = StreamSegMetrics(CONFIG.num_classes)

    loss = [0.0, 0]
    for x, y in tqdm(data):
        with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
            x, y = x.to(CONFIG.device), y.to(CONFIG.device).long()

            if CONFIG.arch == 'deeplabv3_resnet50':
                logits = model(x)

            loss[0] += F.cross_entropy(logits, y, ignore_index=255, reduction='mean')
            loss[1] += x.size(0)
            _, preds = torch.max(logits, 1)
            meter.update(y.cpu().numpy(), preds.cpu().numpy())
    
    mIoU = meter.get_results()['Mean IoU']
    loss = loss[0] / loss[1]
    label = 'NOISY' if inject else 'CLEAN'
    logging.info(f'[{label}] Mean IoU: {100 * mIoU:.2f} - Loss: {loss}')


def train(model: nn.Module, injector: Injector, data: dict):

    # Create optimizers & schedulers
    if CONFIG.arch == 'deeplabv3_resnet50':
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * CONFIG.experiment_args['lr']},
            {'params': model.classifier.parameters(), 'lr': CONFIG.experiment_args['lr']},
        ], lr=CONFIG.experiment_args['lr'], momentum=0.9, weight_decay=CONFIG.experiment_args['wd'])
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=CONFIG.epochs, power=0.9)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Load checkpoint (if it exists)
    cur_epoch = 0
    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])

    # Check multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.to(CONFIG.device)
    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        model.train()
        injector.inject = True
        
        for batch_idx, batch in enumerate(tqdm(data['train'])):
            
            # Compute loss
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                x, y = batch
                x, y = x.to(CONFIG.device), y.to(CONFIG.device).long()
                
                if CONFIG.arch == 'deeplabv3_resnet50':
                    loss = F.cross_entropy(model(x), y, ignore_index=255, reduction='mean') / CONFIG.grad_accum_steps

            # Optimization step
            scaler.scale(loss).backward()

            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        scheduler.step()
        
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch}]')
        evaluate(model, injector, data['test'], inject=False)

        # Save checkpoint
        if not CONFIG.skip_checkpoints:
            checkpoint = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            }
            torch.save(checkpoint, os.path.join(CONFIG.save_dir, 'last.pth'))


def load_cityscapes_dataset():
    
    if CONFIG.dataset == 'cityscapes':
        """ CityScapes Dataset And Augmentation"""
        CONFIG.num_classes = 19

        train_transform = et.ExtCompose([
            #et.ExtResize(512),
            et.ExtRandomCrop(size=(CONFIG.dataset_args['crop_size'], CONFIG.dataset_args['crop_size'])),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=CONFIG.dataset_args['data_root'],
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=CONFIG.dataset_args['data_root'],
                             split='val', transform=val_transform)
        
    elif CONFIG.dataset == 'GTA5':
        """ GTA5 Dataset And Augmentation"""
        CONFIG.num_classes = 19

        train_transform = et.ExtCompose([
            #et.ExtResize(512),
            et.ExtResize(size=(1914, 1052)),
            et.ExtRandomCrop(size=CONFIG.dataset_args['crop_size']),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(size=(1914, 1052)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = GTA5(root=CONFIG.dataset_args['data_root'],
                               split='train', transforms=train_transform)
        val_dst = GTA5(root=CONFIG.dataset_args['data_root'],
                             split='val', transforms=val_transform)
        
    # Dataloaders
    train_loader = SeededDataLoader(
        train_dst,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = SeededDataLoader(
        val_dst,
        batch_size=CONFIG.batch_size//2,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return {'train': train_loader, 'test': test_loader}


def setup_experiment():
    # Load dataset
    data = load_cityscapes_dataset()

    # Load model
    model = eval(CONFIG.arch)(num_classes=CONFIG.num_classes)

    model.to(CONFIG.device)

    # Load Injector
    injector = Injector(CONFIG.experiment_args['error_model'], 
                        CONFIG.experiment_args['p'], 
                        CONFIG.experiment_args['train_aware'])

    # Setup Injections
    modules = tuple([eval(s) for s in CONFIG.experiment_args['injected_modules']])

    injector.hook_model(model, modules=modules)

    for m in model.modules(): # Hook NaNs
        if isinstance(m, modules):
            m.register_forward_hook(lambda m, i, o: torch.nan_to_num(o, 0.0))

    logging.info(CONFIG) # Save config used
    return model, injector, data


def main():

    # Run Training
    if not CONFIG.test_only:
        # Setup Experiment
        model, injector, data = setup_experiment()
        train(model, injector, data)

    # Run Test
    assert os.path.exists(os.path.join(CONFIG.save_dir, 'last.pth')), 'Checkpoint not found.'
    # Setup Experiment
    model, injector, data = setup_experiment()
    logging.info('[TEST]')
    if torch.load(CONFIG.experiment_args['checkpoint_path']) is not None:
        checkpoint = torch.load(CONFIG.experiment_args['checkpoint_path'])
    else:
        torch.load(os.path.join(CONFIG.save_dir, 'last.pth'))
    model.load_state_dict(checkpoint['model'])

    # Check multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.to(CONFIG.device)

    # Test
    evaluate(model, injector, data['test'], inject=True)
    evaluate(model, injector, data['test'], inject=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    # Parse arguments
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Setup output directory
    CONFIG.save_dir = os.path.join('/data/neutrons/record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, 'log.txt'), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()