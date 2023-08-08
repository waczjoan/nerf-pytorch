from nerf_pytorch.trainers.Trainer import Trainer
from nerf_pytorch.load_blender import load_blender_data


class BlenderTrainer(Trainer):
    def __init__(
        self,
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
        i_print,
        **kwargs
    ):

        self.half_res = half_res
        self.testskip = testskip
        self.white_bkgd = white_bkgd

        self.near = 2.
        self.far = 6.

        super().__init__(
            dataset_type=dataset_type,
            render_test=render_test,
            basedir=basedir,
            expname=expname,
            config_path=config_path,
            device=device,
            render_factor=render_factor,
            chunk=chunk,
            N_rand=N_rand,
            no_batching=no_batching,
            datadir=datadir,
            multires=multires,
            i_embed=i_embed,
            multires_views=multires_views,
            netchunk=netchunk,
            lrate=lrate,
            lrate_decay=lrate_decay,
            use_viewdirs=use_viewdirs,
            N_importance=N_importance,
            netdepth=netdepth,
            netwidth=netwidth,
            netdepth_fine=netdepth_fine,
            netwidth_fine=netwidth_fine,
            ft_path=ft_path,
            perturb=perturb,
            raw_noise_std=raw_noise_std,
            N_samples=N_samples,
            lindisp=lindisp,
            precrop_iters=precrop_iters,
            precrop_frac=precrop_frac,
            i_weights=i_weights,
            i_testset=i_testset,
            i_video=i_video,
            i_print=i_print,
            **kwargs
        )

    def load_data(self):
        images, poses, render_poses, hwf, i_split = load_blender_data(
            self.datadir, self.half_res, self.testskip
        )
        print('Loaded blender', images.shape, render_poses.shape, hwf, self.datadir)
        i_train, i_val, i_test = i_split

        if self.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

        return hwf, poses, i_test, i_val, i_train, images