from tqdm import trange


from nerf_pytorch.trainers.LLF import LLFTrainer
from nerf_pytorch.trainers.Blender import BlenderTrainer
from nerf_pytorch.trainers.deepvoxels import DeepvoxelsTrainer
from nerf_pytorch.trainers.Linemod import LinemodTrainer
from nerf_pytorch.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    dataset_type,
    basedir,
    expname,
    config_path,
    device,
    no_batching,
    half_res,
    datadir,
    N_rand=32*32*4,
    white_bkgd=True,
    render_test=False,
    render_only=False,
    testskip=8,
    chunk=1024*32,
    render_factor=0,
    multires=10,
    i_embed=0,
    multires_views=4,
    netchunk=1024*64,
    lrate=5e-4,
    lrate_decay=250,
    use_viewdirs=True,
    N_importance=0,
    netdepth=8,
    netwidth=256,
    netdepth_fine=8,
    netwidth_fine=256,
    ft_path=None,
    perturb=1.0,
    raw_noise_std=0.0,
    N_samples=64,
    lindisp=True,
    precrop_iters=0,
    precrop_frac=0.5,
    i_weights=10000,
    i_testset=5000,
    i_video=5000,
    i_print=100
):

    if dataset_type == "llff":
        trainer = LLFTrainer()
    elif dataset_type == "blender":
        trainer = BlenderTrainer(
            dataset_type,
            render_test,
            basedir,
            expname,
            config_path,
            device,
            render_factor,
            chunk,
            N_rand,
            no_batching,
            half_res,
            testskip,
            white_bkgd,
            datadir,
            multires,
            i_embed,
            multires_views,
            netchunk,
            lrate,
            lrate_decay,
            use_viewdirs,
            N_importance,
            netdepth,
            netwidth,
            netdepth_fine,
            netwidth_fine,
            ft_path,
            perturb,
            raw_noise_std,
            N_samples,
            lindisp,
            precrop_iters,
            precrop_frac,
            i_weights,
            i_testset,
            i_video,
            i_print
        )
    elif dataset_type == "LINEMOD":
        trainer = LinemodTrainer() #TODO
    elif dataset_type == "deepvoxels":
        trainer = DeepvoxelsTrainer() #TODO
    else:
        raise f'Unknown dataset type {dataset_type} exiting'

    hwf, poses, i_test, i_val, i_train, images = trainer.load_data()
    render_poses = None

    if trainer.render_test:
        render_poses = np.array(poses[i_test])
        render_poses = torch.Tensor(render_poses).to(trainer.device)

    hwf = trainer.cast_intrinsics_to_right_types(hwf=hwf)
    trainer.create_log_dir_and_copy_the_config_file()
    optimizer, render_kwargs_train, render_kwargs_test = trainer.create_nerf_model()

    if render_only:
        trainer.render(render_test, images, i_test, render_poses, hwf, render_kwargs_test)
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
            optimizer, render_kwargs_train,
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
            psnr, render_kwargs_train, render_kwargs_test
        )

        trainer.global_step += 1


if __name__=='__main__':
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(
        dataset_type=args.dataset_type,
        render_test=args.render_test,
        render_only=args.render_only,
        basedir=args.basedir,
        expname=args.expname,
        config_path=args.config_path,
        device=device,
        render_factor=args.render_factor,
        chunk=args.chunk,
        N_rand=args.N_rand,
        no_batching=args.no_batching,
        half_res=args.half_res,
        testskip=args.testskip,
        white_bkgd=args.white_bkgd,
        datadir=args.datadir,
        multires=args.multires,
        i_embed=args.i_embed,
        multires_views=args.multires_views,
        netchunk=args.netchunk,
        lrate=args.lrate,
        lrate_decay=args.lrate_decay,
        use_viewdirs=args.use_viewdirs,
        N_importance=args.N_importance,
        netdepth=args.netdepth,
        netwidth=args.netwidth,
        netdepth_fine=args.netdepth_fine,
        netwidth_fine=args.netwidth_fine,
        ft_path=args.ft_path,
        perturb=args.perturb,
        raw_noise_std=args.raw_noise_std,
        N_samples=args.N_samples,
        lindisp=args.lindisp,
        precrop_iters=args.precrop_iters,
        precrop_frac=args.precrop_frac,
        i_weights=args.i_weights,
        i_testset=args.i_testset,
        i_video=args.i_video,
        i_print=args.i_print
    )
