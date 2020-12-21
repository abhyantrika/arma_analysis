"""Main training script for models."""

import argparse

import pytorch_generative as pg


MODEL_DICT = {
    "gated_pixel_cnn": pg.models.gated_pixel_cnn,
    "image_gpt": pg.models.image_gpt,
    "made": pg.models.made,
    "nade": pg.models.nade,
    "pixel_cnn": pg.models.pixel_cnn,
    "pixel_snail": pg.models.pixel_snail,
    # "vae": pg.models.vae,
    # "vd_vae": pg.models.vd_vae,
    # "vq_vae": pg.models.vq_vae,
    # "vq_vae_2": pg.models.vq_vae_2,
}

from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler

from pytorch_generative import datasets
from pytorch_generative import models
from pytorch_generative import trainer
import torch,cv2,os


def main(args):
    device = "cuda" #if args.use_cuda else "cpu"
    # MODEL_DICT[args.model].reproduce(
    #     args.n_epochs, args.batch_size, args.log_dir, device
    # )

    train_loader, test_loader = datasets.get_mnist_loaders(
        args.batch_size, dynamically_binarize=True
    )

    model = models.PixelCNN(
        in_channels=1,
        out_channels=1,
        n_residual=15,
        residual_channels=16,
        head_channels=32,
    )

    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)

    sample_fn = model.sample

    tensor = sample_fn(model)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)


    tensor = torch.squeeze(tensor)

    for i in range(len(tensor)):
        img = tensor[i].detach().cpu().numpy()
        #breakpoint()
        cv2.imwrite(args.save_folder+'/'+str(i)+'.png',img*255)
        print(str(i),' saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the available models to train",
        default="nade",
        choices=list(MODEL_DICT.keys()),
    )
    parser.add_argument(
        "--n-epochs", type=int, help="number of training epochs", default=457
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="the training and evaluation batch_size",
        default=256,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="the directory where to log data",
        default="tmp/run",
    )
    parser.add_argument("--use-cuda", help="whether to use CUDA", action="store_true")

    parser.add_argument("--ckpt", help="path to ckpt",type=str,default='')
    parser.add_argument("--save_folder", help="out_images",type=str,default='out_images')

    args = parser.parse_args()

    main(args)
