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
        weights = torch.exp(-rays_density * deltas)

        # TODO (4): Compute weight used for rendering from transmittance and density
        weights[..., 1:] *= weights[..., :-1].clone()

        return weights

    def _aggregate(
            self,
            weights: torch.Tensor,
            rays_feature: torch.Tensor
    ):
        # TODO (4): Aggregate (weighted sum of) features using weights
        print("weights", weights.shape)
        print("rays_feature", rays_feature.shape)
        feature = torch.sum(weights * rays_feature, dim=1)
        print("feature", feature.shape)

        return feature

    def forward(
            self,
            sampler,
            implicit_fn,
            ray_bundle,
    ):
        B = ray_bundle.shape[0]
        print("B", B)

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start + self._chunk_size]

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
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            )

            # TODO (4): Render (color) features using weights
            feature = self._aggregate(weights, feature)

            # TODO (4): Render depth map
            weights = weights[:, :100, :]
            depth_values = depth_values[:, :100]
            print("depth_values.shape", depth_values.shape)
            print("delta.shape", deltas.shape)
            depth = self._aggregate(weights, depth_values)
            print("depth.shape", depth.shape)
            exit(0)

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
