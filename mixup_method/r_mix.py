import matplotlib.pyplot as plt
import torch
import torch.functional as F
import torch.nn as nn
import wandb
from torch.autograd import Variable
from .constants import labels_name
from .utils import inverse_normalize


class RMix:
    def __call__(self, inputs, targets, percentiles, args):
        top_k_patch = torch.as_tensor(percentiles, device="cuda").float()
        with torch.no_grad():
            if args.use_random_patches:
                mixed_inputs, mixed_targets = mix_topleast_choose_top(
                    inputs,
                    targets,
                    self.mixup_saliency,
                    top_k_patch,
                    args,
                    self.config,
                    self.config["size"] // self.mixup_saliency.shape[-1],
                    self.episode_counter,
                    self.vis_flag,
                )
            else:
                mixed_inputs, mixed_targets = mix_topleast_choose_top(
                    inputs,
                    targets,
                    self.saliency,
                    top_k_patch,
                    args,
                    self.config,
                    self.patch_size,
                    self.episode_counter,
                    self.vis_flag,
                )
        # Train the model
        mixed_inputs, mixed_targets = Variable(
            mixed_inputs, requires_grad=True
        ), Variable(mixed_targets)
        outputs = self.model(mixed_inputs.float())

        loss = torch.mean(
            torch.sum(-mixed_targets * nn.LogSoftmax(-1)(outputs), dim=1)
        )
        
        return None, mixed_inputs, outputs, loss
            
def mix_topleast_choose_top(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    saliency: torch.Tensor,
    top_k_patch: torch.Tensor,
    args,
    config: dict,
    patch_size: int,
    episode: int,
    vis_flag=False,
):
    """
    Input Mixup a batch on patch level
    Method with one top_k_patch for each image

    x: input batch, shape (B, 3, W, H)
    y: scalar batch labels, shape (B,)
    saliency: saliency of x, shape (B, patch_size, patch_size)
    top_k_patch: vector in range [0.0, 1.0]. Top-k patch values for each image in the batch
    args, config: objects
    patch_size: int (default 16). Size of each patch.
    episode: current epoch to visualize
    vis_flat: bool, whether to visualize the inputs.
    """
    inputs_norm = inputs.clone()
    saliency_mixup = saliency.clone()
    batch_size = inputs.shape[0]
    one_hot_y = F.one_hot(labels, config["num_class"])

    # Sample weights
    w1, w2 = torch.distributions.dirichlet.Dirichlet(
        torch.as_tensor(
            [args.dirichlet_alpha, args.dirichlet_alpha], device="cuda"
        ).float()
    ).sample()
    saliency_flat = torch.reshape(saliency_mixup, (batch_size, -1))  # Flatten
    quants = torch.quantile(
        saliency_flat, top_k_patch, dim=1, keepdim=True
    )  # Calculate quantile
    quants = quants[
        torch.arange(batch_size), torch.arange(batch_size), 0
    ]  # Construct a matrix and get the diag
    mask = (saliency_mixup >= quants[:, None, None]).type(
        torch.float64
    )  # Construct mask, shape: (B, W, H)
    mask = mask.unsqueeze(1)  # Add channel dimension

    # Upsample mask
    mask = F.interpolate(mask.float(), scale_factor=patch_size, mode="nearest")

    index = torch.randperm(batch_size).cuda()
    mask_perm = mask[index, :]
    inputs_perm = inputs_norm[index, :]
    one_hot_y_perm = one_hot_y[index, :]

    mask_inter = torch.where(mask == mask_perm, 1, 0)  # Case 1 and 3
    mask_neq = torch.where(mask != mask_perm, 1, 0)  # Case 2

    inputs_eq = mask_inter * (w1 * inputs_norm + w2 * inputs_perm)  # Mixup case 1 and 3
    inputs_neq = mask_neq * (
        mask * inputs_norm + mask_perm * inputs_perm
    )  # Ignore case 2

    mixed_inputs = inputs_neq + inputs_eq  # Combine results

    mixup_weight = (
        (mask_inter.flatten(1).sum(1) / (config["size"] * config["size"]))
        .unsqueeze(1)
        .repeat(1, config["num_class"])
    )
    x1_weight = (
        ((mask_neq * mask).flatten(1).sum(1) / (config["size"] * config["size"]))
        .unsqueeze(1)
        .repeat(1, config["num_class"])
    )
    x2_weight = (
        ((mask_neq * mask_perm).flatten(1).sum(1) / (config["size"] * config["size"]))
        .unsqueeze(1)
        .repeat(1, config["num_class"])
    )
    mixed_labels = (
        mixup_weight * (w1 * one_hot_y + w2 * one_hot_y_perm)
        + x1_weight * one_hot_y
        + x2_weight * one_hot_y_perm
    )

    # For visualization
    if vis_flag:
        inputs = inverse_normalize(inputs.clone(), dataset=args.dataset)
        saliency_vis = saliency_mixup.clone()
        inputs_eq_vis = mask_inter[0] * (
            w1 * inputs[0] + w2 * inputs[index, :][0]
        )  # Mixup case 1 and 3
        inputs_neq_vis = mask_neq[0] * (
            mask[0] * inputs[0] + mask_perm[0] * inputs[index, :][0]
        )  # Ignore case 2
        mixed_inputs_vis = inputs_neq_vis + inputs_eq_vis  # Combine results

        fig = plt.figure(figsize=(20, 7))
        spec = fig.add_gridspec(2, 6)
        img1 = fig.add_subplot(spec[0, 0])
        img1.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
        img1.set_title(
            {
                x: y
                for x, y in dict(zip(labels_name, one_hot_y[0].cpu().numpy())).items()
                if y != 0.0
            }
        )

        sal1 = fig.add_subplot(spec[0, 1])
        sal1.imshow(saliency_vis[0].cpu().numpy())
        sal1.set_title("Saliency map")

        topk1 = fig.add_subplot(spec[0, 2])
        topk1.imshow(mask[0].permute(1, 2, 0).squeeze(-1).cpu().numpy())
        topk1.set_title("Top-k: " + str(top_k_patch[0].cpu().numpy()))

        mixregion = fig.add_subplot(spec[0, 3])
        mixregion.imshow(inputs_eq_vis.permute(1, 2, 0).squeeze(-1).cpu().numpy())
        mixregion.set_title("Mix region")

        img2 = fig.add_subplot(spec[1, 0])
        img2.imshow(inputs[index, :][0].permute(1, 2, 0).cpu().numpy())
        img2.set_title(
            {
                x: y
                for x, y in dict(
                    zip(labels_name, one_hot_y_perm[0].cpu().numpy())
                ).items()
                if y != 0.0
            }
        )

        sal2 = fig.add_subplot(spec[1, 1])
        sal2.imshow(saliency_vis[index, :][0].cpu().numpy())
        sal2.set_title("Saliency map")

        topk2 = fig.add_subplot(spec[1, 2])
        topk2.imshow(mask[index, :][0].permute(1, 2, 0).squeeze(-1).cpu().numpy())
        topk2.set_title("Top-k: " + str(top_k_patch[index][0].cpu().numpy()))

        nonmixregion = fig.add_subplot(spec[1, 3])
        nonmixregion.imshow(inputs_neq_vis.permute(1, 2, 0).squeeze(-1).cpu().numpy())
        nonmixregion.set_title("Non-Mix region")

        mixedimg = fig.add_subplot(spec[:, 4:])
        mixedimg.imshow(mixed_inputs_vis.permute(1, 2, 0).cpu().numpy())
        mixedimg.set_title(
            str(
                {
                    x: y
                    for x, y in dict(
                        zip(labels_name, mixed_labels[0].cpu().numpy())
                    ).items()
                    if y != 0.0
                }
            )
            + f", epoch{episode}"
        )
        if args.use_wandb:
            wandb.log({"vis": plt})
        plt.savefig(f"./vis/Visualization_Epoch{episode}.png")

    return mixed_inputs, mixed_labels

def mix_topleast_choose_least(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    saliency: torch.Tensor,
    top_k_patch: torch.Tensor,
    args,
    config: dict,
    patch_size: int,
    episode: int,
    vis_flag=False,
):
    """
    Input Mixup a batch on patch level
    Method with one top_k_patch for each image

    x: input batch, shape (B, 3, W, H)
    y: scalar batch labels, shape (B,)
    saliency: saliency of x, shape (B, patch_size, patch_size)
    top_k_patch: vector in range [0.0, 1.0]. Top-k patch values for each image in the batch
    args, config: objects
    patch_size: int (default 16). Size of each patch.
    episode: current epoch to visualize
    vis_flat: bool, whether to visualize the inputs.
    """
    inputs_norm = inputs.clone()
    saliency_mixup = saliency.clone()
    batch_size = inputs.shape[0]
    one_hot_y = F.one_hot(labels, config["num_class"])

    # Sample weights
    w1, w2 = torch.distributions.dirichlet.Dirichlet(
        torch.as_tensor(
            [args.dirichlet_alpha, args.dirichlet_alpha], device="cuda"
        ).float()
    ).sample()
    saliency_flat = torch.reshape(saliency_mixup, (batch_size, -1))  # Flatten
    quants = torch.quantile(
        saliency_flat, top_k_patch, dim=1, keepdim=True
    )  # Calculate quantile
    quants = quants[
        torch.arange(batch_size), torch.arange(batch_size), 0
    ]  # Construct a matrix and get the diag
    mask = (saliency_mixup >= quants[:, None, None]).type(
        torch.float64
    )  # Construct mask, shape: (B, W, H)
    mask = mask.unsqueeze(1)  # Add channel dimension

    # Upsample mask
    mask = F.interpolate(mask.float(), scale_factor=patch_size, mode="nearest")

    index = torch.randperm(batch_size).cuda()
    mask_perm = mask[index, :]
    inputs_perm = inputs_norm[index, :]
    one_hot_y_perm = one_hot_y[index, :]

    mask_inter = torch.where(mask == mask_perm, 1, 0)  # Case 1 and 3
    mask_neq = torch.where(mask != mask_perm, 1, 0)  # Case 2

    inputs_eq = mask_inter * (w1 * inputs_norm + w2 * inputs_perm)  # Mixup case 1 and 3
    inputs_neq = mask_neq * (
        torch.logical_not(mask) * inputs_norm + torch.logical_not(mask_perm) * inputs_perm
    )  # Ignore case 2

    mixed_inputs = inputs_neq + inputs_eq  # Combine results

    mixup_weight = (
        (mask_inter.flatten(1).sum(1) / (config["size"] * config["size"]))
        .unsqueeze(1)
        .repeat(1, config["num_class"])
    )
    x1_weight = (
        ((mask_neq * mask).flatten(1).sum(1) / (config["size"] * config["size"]))
        .unsqueeze(1)
        .repeat(1, config["num_class"])
    )
    x2_weight = (
        ((mask_neq * mask_perm).flatten(1).sum(1) / (config["size"] * config["size"]))
        .unsqueeze(1)
        .repeat(1, config["num_class"])
    )
    mixed_labels = (
        mixup_weight * (w1 * one_hot_y + w2 * one_hot_y_perm)
        + x1_weight * one_hot_y
        + x2_weight * one_hot_y_perm
    )

    # For visualization
    if vis_flag:
        inputs = inverse_normalize(inputs.clone(), dataset=args.dataset)
        saliency_vis = saliency_mixup.clone()
        inputs_eq_vis = mask_inter[0] * (
            w1 * inputs[0] + w2 * inputs[index, :][0]
        )  # Mixup case 1 and 3
        inputs_neq_vis = mask_neq[0] * (
            mask[0] * inputs[0] + mask_perm[0] * inputs[index, :][0]
        )  # Ignore case 2
        mixed_inputs_vis = inputs_neq_vis + inputs_eq_vis  # Combine results

        fig = plt.figure(figsize=(20, 7))
        spec = fig.add_gridspec(2, 6)
        img1 = fig.add_subplot(spec[0, 0])
        img1.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
        img1.set_title(
            {
                x: y
                for x, y in dict(zip(labels_name, one_hot_y[0].cpu().numpy())).items()
                if y != 0.0
            }
        )

        sal1 = fig.add_subplot(spec[0, 1])
        sal1.imshow(saliency_vis[0].cpu().numpy())
        sal1.set_title("Saliency map")

        topk1 = fig.add_subplot(spec[0, 2])
        topk1.imshow(mask[0].permute(1, 2, 0).squeeze(-1).cpu().numpy())
        topk1.set_title("Top-k: " + str(top_k_patch[0].cpu().numpy()))

        mixregion = fig.add_subplot(spec[0, 3])
        mixregion.imshow(inputs_eq_vis.permute(1, 2, 0).squeeze(-1).cpu().numpy())
        mixregion.set_title("Mix region")

        img2 = fig.add_subplot(spec[1, 0])
        img2.imshow(inputs[index, :][0].permute(1, 2, 0).cpu().numpy())
        img2.set_title(
            {
                x: y
                for x, y in dict(
                    zip(labels_name, one_hot_y_perm[0].cpu().numpy())
                ).items()
                if y != 0.0
            }
        )

        sal2 = fig.add_subplot(spec[1, 1])
        sal2.imshow(saliency_vis[index, :][0].cpu().numpy())
        sal2.set_title("Saliency map")

        topk2 = fig.add_subplot(spec[1, 2])
        topk2.imshow(mask[index, :][0].permute(1, 2, 0).squeeze(-1).cpu().numpy())
        topk2.set_title("Top-k: " + str(top_k_patch[index][0].cpu().numpy()))

        nonmixregion = fig.add_subplot(spec[1, 3])
        nonmixregion.imshow(inputs_neq_vis.permute(1, 2, 0).squeeze(-1).cpu().numpy())
        nonmixregion.set_title("Non-Mix region")

        mixedimg = fig.add_subplot(spec[:, 4:])
        mixedimg.imshow(mixed_inputs_vis.permute(1, 2, 0).cpu().numpy())
        mixedimg.set_title(
            str(
                {
                    x: y
                    for x, y in dict(
                        zip(labels_name, mixed_labels[0].cpu().numpy())
                    ).items()
                    if y != 0.0
                }
            )
            + f", epoch{episode}"
        )
        if args.use_wandb:
            wandb.log({"vis": plt})
        plt.savefig(f"./vis/Visualization_Epoch{episode}.png")

    return mixed_inputs, mixed_labels