import numpy as np
import time
import torch
from tqdm import trange


from nerf_pytorch.trainers.LLF import LLFTrainer
from nerf_pytorch.trainers.Blender import BlenderTrainer
from nerf_pytorch.trainers.deepvoxels import DeepvoxelsTrainer
from nerf_pytorch.trainers.Linemod import LinemodTrainer
from nerf_pytorch.utils import *

#def train(
#    dataset_type,
#    render_only,
#    render_test
#):


def train():
    parser = config_parser()
    args = parser.parse_args()
    dataset_type = args.dataset_type
    render_only = args.render_only
    render_test = args.render_test

    if dataset_type == "llff":
        trainer = LLFTrainer()
    elif dataset_type == "blender":
        trainer = BlenderTrainer(
            dataset_type,
            render_test,
            args.basedir,
            args.expname,
            args.config_path,
            'cpu',
            args.render_factor,
            args.chunk,
            args.N_rand,
            args.no_batching,
            args.half_res,
            args.testskip,
            args.white_bkgd,
            args.datadir
        )
    elif dataset_type == "LINEMOD":
        trainer = LinemodTrainer()
    elif dataset_type == "deepvoxels":
        trainer = DeepvoxelsTrainer()
    else:
        raise f'Unknown dataset type {dataset_type} exiting'

    hwf, poses, i_test, i_val, i_train, images = trainer.load_data()
    if trainer.render_test:
        render_poses = np.array(poses[i_test])
        render_poses = torch.Tensor(render_poses).to(trainer.device)

    hwf = trainer.cast_intrinsics_to_right_types(hwf=hwf)
    trainer.create_log_dir_and_copy_the_config_file()
    optimizer = trainer.create_nerf_model()

    if render_only:
        trainer.render(render_test, images, i_test, render_poses, hwf)
        return render_only

    images, poses, rays_rgb, i_batch = trainer.prepare_raybatch_tensor_if_batching_random_rays(
        poses, images, i_train
    )

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = trainer.start + 1
    for i in trange(start, N_iters):
        rays_rgb, i_batch, batch_rays, target_s = trainer.sample_random_ray_batch(
            rays_rgb,
            i_batch,
            i_train,
            images,
            poses,
            i
        )

        trans, loss, psnr, psnr0 = trainer.core_optimization_loop(
            optimizer,
            batch_rays, i, target_s,
        )

        trainer.update_learning_rate(optimizer)

        trainer.rest_is_logging(
            i,
            render_poses,
            hwf,
            poses,
            i_test,
            images,
            loss,
            psnr
        )

        trainer.global_step += 1


if __name__=='__main__':
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
