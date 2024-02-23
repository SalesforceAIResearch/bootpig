#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


from omegaconf import OmegaConf
import copy
import hydra
import logging
import math
import os
import shutil

from accelerate import DistributedDataParallelKwargs
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from transformers import CLIPTextModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.optimization import get_scheduler


from data import GenDataset
import json
from model import BootPIGUNet
from model.utils import torch_dfs, AttnProcessor2_0_withScaling


logger = get_logger(__name__)


def save_pipeline(base_unet, ref_unet, args, accelerator, weight_dtype, save_path):
    ref_unet = accelerator.unwrap_model(ref_unet)
    base_unet = accelerator.unwrap_model(base_unet)
    ref_unet.save_pretrained(os.path.join(save_path, "ref_unet"))
    base_unet.save_pretrained(os.path.join(save_path, "base_unet"))


def collate_fn(examples):
    pixel_values = torch.stack([example["instance_images"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = [
        torch.cat(example["reference_images"]) for example in examples
    ]
    conditioning_pixel_values = torch.stack(conditioning_pixel_values)
    conditioning_pixel_values = conditioning_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()

    input_ids = torch.cat([example["instance_prompt_ids"] for example in examples])
    ref_input_ids = torch.cat(
        [example["instance_refprompt_ids"] for example in examples]
    )

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "ref_input_ids": ref_input_ids,
    }


@hydra.main(version_base=None, config_path="./configs/training/", config_name="bootpig")
def main(origargs):
    args = copy.deepcopy(origargs)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.logging.output_dir, logging_dir=args.logging.log_dir
    )
    print("Using find_unused_parameters={}".format(args.data.empty_ref_p > 0))
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=args.data.empty_ref_p > 0
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.optim.gradient_accumulation_steps,
        mixed_precision=args.exec.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.exec.seed is not None:
        set_seed(args.exec.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.logging.run_dir, exist_ok=True)
        args_dict = OmegaConf.to_container(args, resolve=True)
        with open(os.path.join(args.logging.run_dir, "train_args.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

    # Load the tokenizer
    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, revision=args.model.revision, use_fast=False
        )
    elif args.model.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.model.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = CLIPTextModel

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.model.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.model.revision,
    )
    base_unet = BootPIGUNet.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.model.revision,
        bootpig_mode="base",
    )
    ref_unet = BootPIGUNet.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.model.revision,
        bootpig_mode="ref",
    )

    attn_modules = [
        module
        for module in torch_dfs(base_unet)
        if isinstance(module, BasicTransformerBlock)
    ]
    for mi, mod in enumerate(attn_modules):
        mod.attn1.set_processor(
            AttnProcessor2_0_withScaling(ref_scale=args.exec.train_ref_scale)
        )

    vae.requires_grad_(False)
    base_unet.train()
    base_unet.requires_grad_(False)
    for n, p in base_unet.named_parameters():
        if "attn1" in n:
            p.requires_grad_(True)

    text_encoder.requires_grad_(False)
    ref_unet.train()

    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(ref_unet).dtype != torch.float32:
        raise ValueError(
            f"ref_unet loaded as datatype {accelerator.unwrap_model(ref_unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.exec.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = []
    params_to_optimize.extend(
        [
            {
                "params": list(ref_unet.parameters()),
                "lr": args.optim.ref_unet_learning_rate,
            }
        ]
    )
    attn_modules = [
        module
        for module in torch_dfs(base_unet)
        if isinstance(module, BasicTransformerBlock)
    ]
    for i, module in enumerate(attn_modules):
        if not module.only_cross_attention:
            params_to_optimize.append(
                {
                    "params": [p for n, p in module.named_parameters() if "attn1" in n],
                    "lr": args.optim.base_unet_learning_rate,
                }
            )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.optim.ref_unet_learning_rate,
        betas=(args.optim.adam_beta1, args.optim.adam_beta2),
        weight_decay=args.optim.adam_weight_decay,
        eps=args.optim.adam_epsilon,
    )
    image_norm_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    print("Loading SD Generated Data...")
    train_dataset = GenDataset(
        args.data.data_dir,
        tokenizer,
        image_norm_transform=image_norm_transform,
        refimage_norm_transform=image_norm_transform,
        resolution=args.data.resolution,
        empty_prompt_p=args.data.empty_prompt_p,
        jitter_p=args.data.jitter_p,
        erase_p=args.data.erase_p,
        min_pct_area=args.data.min_pct_area,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.optim.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.exec.dataloader_num_workers,
        drop_last=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.optim.gradient_accumulation_steps
    )
    if args.optim.max_steps is None:
        args.optim.max_steps = args.optim.epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # args.optim.max_steps *= len(train_dataset)
    lr_scheduler = get_scheduler(
        args.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.optim.lr_warmup_steps
        * args.optim.gradient_accumulation_steps,
        num_training_steps=args.optim.max_steps
        * args.optim.gradient_accumulation_steps,
        num_cycles=args.optim.lr_num_cycles,
        power=args.optim.lr_power,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Prepare everything with our `accelerator`.
    base_unet = accelerator.prepare(base_unet)
    ref_unet = accelerator.prepare(ref_unet)
    (
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Move vae, base_unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.optim.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.optim.max_steps = args.optim.epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.optim.epochs = math.ceil(args.optim.max_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.logging.wandb_project,
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs={
                "wandb": {
                    "name": args.logging.name,
                    "entity": args.logging.wandb_username,
                    "dir": args.logging.log_dir,
                }
            },
        )

    # Train!
    total_batch_size = (
        args.optim.train_batch_size
        * accelerator.num_processes
        * args.optim.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.optim.epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.optim.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.optim.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {args.optim.max_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.exec.resume_from_checkpoint:
        if args.exec.resume_from_checkpoint != "latest":
            path = os.path.basename(args.exec.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.logging.run_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.exec.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.exec.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.logging.run_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.optim.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.optim.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.optim.max_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, args.optim.epochs):
        for step, batch in enumerate(train_dataloader):
            if (
                args.exec.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.optim.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(ref_unet), accelerator.accumulate(base_unet):
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                ref_encoder_hidden_states = text_encoder(batch["ref_input_ids"])[0]
                # ref_encoder_hidden_states = encoder_hidden_states
                rnd = torch.rand(1).to(device=accelerator.device)
                all_rnds = accelerator.gather(rnd).detach()
                reffeats = None
                ref_pred = None

                # Dropout references
                if not all_rnds[0] < args.data.empty_ref_p:
                    condition_image = batch["conditioning_pixel_values"].to(
                        dtype=weight_dtype
                    )
                    condition_latents = vae.encode(condition_image).latent_dist.sample()
                    condition_latents = condition_latents * vae.config.scaling_factor
                    condition_noise = torch.randn_like(condition_latents)
                    noisy_condition_latents = noise_scheduler.add_noise(
                        condition_latents, condition_noise, timesteps
                    )
                    reffeats = []
                    ref_pred = ref_unet(
                        noisy_condition_latents,
                        timesteps,
                        encoder_hidden_states=ref_encoder_hidden_states,
                        return_dict=True,
                        reffeats=reffeats,
                    )[0]
                    reffeats = [feat.to(dtype=weight_dtype) for feat in reffeats]

                model_pred = base_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    reffeats=reffeats,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Hacky solution to make sure DDP doesn't throw an error
                if ref_pred is not None:
                    loss += 0 * ref_pred.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(ref_unet.parameters())
                    attn_modules = [
                        module
                        for module in torch_dfs(base_unet)
                        if isinstance(module, BasicTransformerBlock)
                    ]
                    for i, module in enumerate(attn_modules):
                        if not module.only_cross_attention:
                            params_to_clip.extend(
                                [
                                    p
                                    for n, p in module.named_parameters()
                                    if "attn1" in n
                                ]
                            )
                    accelerator.clip_grad_norm_(
                        params_to_clip, args.optim.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.exec.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.logging.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.logging.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.logging.run_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )
                            removeable_checkpoints = [
                                ckpt
                                for ckpt in checkpoints
                                if int(ckpt.split("-")[1])
                                % args.logging.checkpoint_keep_freq
                                != 0
                            ]

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if (
                                len(removeable_checkpoints)
                                >= args.logging.checkpoints_total_limit
                            ):
                                num_to_remove = (
                                    len(removeable_checkpoints)
                                    - args.logging.checkpoints_total_limit
                                    + 1
                                )
                                removing_checkpoints = removeable_checkpoints[
                                    0:num_to_remove
                                ]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.logging.run_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.logging.run_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        save_pipeline(
                            base_unet=base_unet,
                            ref_unet=ref_unet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            save_path=os.path.join(
                                args.logging.run_dir,
                                f"checkpoint-{global_step}",
                            ),
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.optim.max_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ref_unet = accelerator.unwrap_model(ref_unet)
        ref_unet.save_pretrained(
            os.path.join(
                args.logging.run_dir,
                f"checkpoint-{global_step}",
                "ref_unet",
            )
        )
        accelerator.unwrap_model(base_unet).save_pretrained(
            os.path.join(
                args.logging.run_dir,
                f"checkpoint-{global_step}",
                "base_unet",
            )
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
