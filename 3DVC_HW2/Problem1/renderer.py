import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
            self,
            cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
            self,
            deltas,
            rays_density: torch.Tensor,
            eps: float = 1e-10
    ):
        # TODO (4): Compute transmittance using the equation described in the README
        t = torch.exp(-rays_density * deltas)
        T = torch.concat((torch.ones(size=(t.shape[0], 1, t.shape[2]), device=t.device),
                          torch.cumprod(t, dim=1)[:, :-1, :]), dim=1)

        # TODO (4): Compute weight used for rendering from transmittance and density
        weights = T * (1 - t)

        return weights, T, t

    def _aggregate(
            self,
            weights: torch.Tensor,
            rays_feature: torch.Tensor
    ):
        # TODO (4): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def forward(
            self,
            sampler,
            implicit_fn,
            ray_bundle,
    ):
        B = ray_bundle.shape[0]
        print("\nB", B)

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            print("chunk_start", chunk_start)
            cur_ray_bundle = ray_bundle[chunk_start : min(chunk_start + self._chunk_size, B)]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights, T, t = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            )

            # TODO (4): Render (color) features using weights
            feature = self._aggregate(weights, feature.view(-1, n_pts, 3))

            # TODO (4): Render depth map
            depth = torch.inf * torch.ones(size=(weights.shape[0],), device=weights.device)

            # # method 1
            # index = torch.argmin(T, dim=1).squeeze()
            # T = T[torch.arange(depth_values.shape[0]), index, 0]
            # select_cond = T < 0.02
            # index = index[select_cond]
            # depth[select_cond] = depth_values[torch.arange(depth_values.shape[0])[select_cond], index]

            # method 2
            T[T >= 1] = -torch.inf
            select_cond = T != -torch.inf
            t /= torch.sum(torch.mul(t, select_cond), dim=1, keepdim=True)
            t[~select_cond] = 0
            t = t.squeeze()
            depth = self._aggregate(t, depth_values)
            depth[depth == 0] = torch.inf

            # depth = self._aggregate(1 - T, depth_values)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
                [chunk_out[k] for chunk_out in chunk_outputs],
                dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
