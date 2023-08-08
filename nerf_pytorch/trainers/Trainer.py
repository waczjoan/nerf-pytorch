from pathlib import Path

from nerf_pytorch.utils import *


class Trainer:

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
    ):
        self.start = None
        self.dataset_type = dataset_type
        self.render_test = render_test
        self.basedir = basedir
        self.expname = expname
        self.config_path = config_path
        self.device = device
        self.chunk = chunk
        self.render_factor = render_factor
        self.N_rand = N_rand
        self.no_batching = no_batching
        self.use_batching = not self.no_batching
        self.datadir = datadir
        self.multires = multires
        self.i_embed = i_embed,
        self.multires_views = multires_views
        self.netwidth_fine = netwidth_fine
        self.netchunk = netchunk
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.use_viewdirs = use_viewdirs
        self.N_importance = N_importance
        self.netdepth = netdepth
        self.netwidth = netwidth
        self.netdepth_fine = netdepth_fine
        self.netwidth_fine = netwidth_fine
        self.ft_path = ft_path
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.N_samples = N_samples
        self.lindisp = lindisp
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac
        self.i_weights = i_weights
        self.i_testset = i_testset
        self.i_video = i_video
        self.i_print = i_print

        self.K = None
        self.global_step = None
        self.W = None
        self.H = None

    def load_data(
        self,
        datadir: Path
    ):
        """Load data and prepare poses."""
        pass

    def cast_intrinsics_to_right_types(self, hwf):
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if self.K is None:
            self.K = np.array(
                [[focal, 0, 0.5 * W],
                 [0, focal, 0.5 * H],
                 [0, 0, 1]]
            )

        self.H = H
        self.W = W
        return hwf

    def create_log_dir_and_copy_the_config_file(self):
        basedir = self.basedir
        expname = self.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            dict = self.__dict__
            for arg in dict:
                file.write('{} = {}\n'.format(arg, dict[arg]))
        if self.config_path is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.config_path, 'r').read())

    def create_nerf_model(self):
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(self)
        self.global_step = start
        self.start = start

        bds_dict = {
            'near': self.near,
            'far': self.far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        return optimizer, render_kwargs_train, render_kwargs_test

    def render(self, render_test, images, i_test, render_poses, hwf, render_kwargs_test):
        with torch.no_grad():
            if render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                self.basedir,
                self.expname,
                'renderonly_{}_{:06d}'.format(
                    'test' if render_test else 'path', self.global_step
                )
            )

            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(
                render_poses, hwf, self.K, self.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=self.render_factor
            )
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

    def prepare_raybatch_tensor_if_batching_random_rays(
        self,
        poses,
        images,
        i_train
    ):
        i_batch = None
        rays_rgb = None

        if self.use_batching :
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
            print('done, concats')
            rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_rgb)

            print('done')
            i_batch = 0

        # Move training data to GPU
        if self.use_batching:
            images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        if self.use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(device)

        return images, poses, rays_rgb, i_batch

    def rest_is_logging(
            self, i, render_poses, hwf, poses, i_test, images, loss, psnr, render_kwargs_train, render_kwargs_test
    ):
        if i % self.i_weights == 0:
            path = os.path.join(self.basedir, self.expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': self.global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % self.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, self.K, self.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_{:06d}_'.format(self.expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % self.i_testset == 0 and i > 0:
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, self.K, self.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % self.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

    def sample_random_ray_batch(
        self,
        rays_rgb,
        i_batch,
        i_train,
        images,
        poses,
        i
    ):
        if self.use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + self.N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += self.N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if self.N_rand is not None:
                rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < self.precrop_iters:
                    dH = int(self.H // 2 * self.precrop_frac)
                    dW = int(self.W // 2 * self.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH),
                            torch.linspace(self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == self.start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {self.precrop_iters}")
                else:
                    coords = torch.stack(
                        torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        return rays_rgb, i_batch, batch_rays, target_s

    def core_optimization_loop(
        self,
        optimizer, render_kwargs_train,
        batch_rays, i, target_s,
    ):
        rgb, disp, acc, extras = render(self.H, self.W, self.K,
            chunk=self.chunk, rays=batch_rays,
            verbose=i < 10, retraw=True,
            **render_kwargs_train
        )

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        return trans, loss, psnr, psnr0

    def update_learning_rate(self, optimizer):
        decay_rate = 0.1
        decay_steps = self.lrate_decay * 1000
        new_lrate = self.lrate * (decay_rate ** (self.global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

    def sample_points(
            self,
            **kwargs
    ):

        z_vals_mid = kwargs["z_vals_mid"]
        weights = kwargs["weights"]
        perturb = kwargs["perturb"]
        pytest = kwargs["pytest"]

        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            self.N_importance,
            det=(perturb == 0.),
            pytest=pytest
        )
        return z_samples
