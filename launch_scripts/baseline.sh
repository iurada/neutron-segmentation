s=0
arch=deeplabv3_resnet50

python main.py \
--seed=${s} \
--experiment_name=${arch}/baseline/run${s} \
--experiment_args='''
{
    "error_model": "block",
    "p": 0.3,
    "train_aware": False,
    "injected_modules": ["nn.Conv2d"],

    "relux_operation": None,

    "lr": 0.01,
    "wd": 0.0001,

    "pretrain": True
}
''' \
--dataset=cityscapes \
--dataset_args='''{"data_root": "data/Cityscapes/", "crop_size": 769}''' \
--arch=${arch} \
--batch_size=6 \
--epochs=100 \
--num_workers=5 \
--grad_accum_steps=1
