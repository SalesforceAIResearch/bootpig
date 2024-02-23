from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect

import numpy as np
import PIL.Image
import torch
from diffusers.configuration_utils import register_to_config
from diffusers import StableDiffusionPipeline
from diffusers.models.attention import BasicTransformerBlock
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import PIL_INTERPOLATION, logging

from diffusers.models import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from .utils import torch_dfs, AttnProcessor2_0_withScaling


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def collect_attn_feats(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
):
    assert "reffeats" in cross_attention_kwargs
    cak_without_ref = {
        k: v
        for k, v in cross_attention_kwargs.items()
        if "reffeats" not in k and "is_cfg" not in k
    }
    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.norm1(
            hidden_states,
            timestep,
            class_labels,
            hidden_dtype=hidden_states.dtype,
        )
    else:
        norm_hidden_states = self.norm1(hidden_states)

    # 1. Self-Attention
    cross_attention_kwargs = (
        cross_attention_kwargs if cross_attention_kwargs is not None else {}
    )
    if self.only_cross_attention:
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cak_without_ref,
        )
    else:
        write_feat = norm_hidden_states
        cross_attention_kwargs["reffeats"].append(write_feat)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cak_without_ref,
        )

    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states

    if self.attn2 is not None:
        norm_hidden_states = (
            self.norm2(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm2(hidden_states)
        )

        # 2. Cross-Attention
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cak_without_ref,
        )
        hidden_states = attn_output + hidden_states

    # 3. Feed-forward
    norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

    ff_output = self.ff(norm_hidden_states)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = ff_output + hidden_states

    return hidden_states


def pass_attn_feats(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
):
    # assert "reffeats" in cross_attention_kwargs
    cak_without_ref = {
        k: v
        for k, v in cross_attention_kwargs.items()
        if k not in ["reffeats", "is_cfg_guidance"]
    }
    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.norm1(
            hidden_states,
            timestep,
            class_labels,
            hidden_dtype=hidden_states.dtype,
        )
    else:
        norm_hidden_states = self.norm1(hidden_states)

    # 1. Self-Attention
    cross_attention_kwargs = (
        cross_attention_kwargs if cross_attention_kwargs is not None else {}
    )
    if self.only_cross_attention or (not "reffeats" in cross_attention_kwargs):
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cak_without_ref,
        )
    else:
        if cross_attention_kwargs["is_cfg_guidance"]:
            num_im_feat = norm_hidden_states.shape[0] * 2 // 3
            attn_output1 = self.attn1(
                norm_hidden_states,  # [:-num_im_feat],
                **cak_without_ref,
            )
            all_ref_features = cross_attention_kwargs["reffeats"].pop(0)
            if not isinstance(all_ref_features, list):
                all_ref_features = [all_ref_features]
            attn_output2 = []
            for ref_features in all_ref_features:
                ref_encoder_hidden_states = torch.cat(
                    [norm_hidden_states[-num_im_feat:], ref_features], dim=1
                )
                attn_output2.append(
                    self.attn1(
                        norm_hidden_states[-num_im_feat:],
                        encoder_hidden_states=ref_encoder_hidden_states,
                        num_ref_feat=ref_features.shape[1],
                        # ref_scale=1.025,
                        **cak_without_ref,
                    )
                    - attn_output1[-num_im_feat:]
                )
            attn_output2 = torch.stack(attn_output2, 0)
            attn_output2_norm = torch.norm(attn_output2, dim=-1, p=2)
            attn_output2_ind = attn_output2_norm.argmax(0, True)
            attn_output2_ind = (
                torch.zeros_like(attn_output2_norm)
                .scatter(0, attn_output2_ind, value=1)
                .unsqueeze(-1)
            )

            attn_output2 = (attn_output2 * attn_output2_ind).sum(0)
            attn_output = attn_output1
            attn_output[-num_im_feat:] = attn_output[-num_im_feat:] + attn_output2
        else:
            all_ref_features = cross_attention_kwargs["reffeats"].pop(0)
            if not isinstance(all_ref_features, list):
                all_ref_features = [all_ref_features]
            attn_output = 0.0
            for ref_features in all_ref_features:
                ref_encoder_hidden_states = torch.cat(
                    [norm_hidden_states, ref_features], dim=1
                )
                attn_output += self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=ref_encoder_hidden_states,
                    num_ref_feat=ref_features.shape[1],
                    **cak_without_ref,
                )
            attn_output /= float(len(all_ref_features))
    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states

    if self.attn2 is not None:
        norm_hidden_states = (
            self.norm2(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm2(hidden_states)
        )

        # 2. Cross-Attention
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cak_without_ref,
        )
        hidden_states = attn_output + hidden_states

    # 3. Feed-forward
    norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

    ff_output = self.ff(norm_hidden_states)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = ff_output + hidden_states

    return hidden_states


class BootPIGUNet(UNet2DConditionModel):
    @register_to_config
    def __init__(
        self,
        sample_size: int | None = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = ...,
        mid_block_type: str | None = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ...,
        only_cross_attention: bool | Tuple[bool] = False,
        block_out_channels: Tuple[int] = ...,
        layers_per_block: int | Tuple[int] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0,
        act_fn: str = "silu",
        norm_num_groups: int | None = 32,
        norm_eps: float = 0.00001,
        cross_attention_dim: int | Tuple[int] = 1280,
        transformer_layers_per_block: int | Tuple[int] | Tuple[Tuple] = 1,
        reverse_transformer_layers_per_block: Tuple[Tuple[int]] | None = None,
        encoder_hid_dim: int | None = None,
        encoder_hid_dim_type: str | None = None,
        attention_head_dim: int | Tuple[int] = 8,
        num_attention_heads: int | Tuple[int] | None = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: str | None = None,
        addition_embed_type: str | None = None,
        addition_time_embed_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1,
        time_embedding_type: str = "positional",
        time_embedding_dim: int | None = None,
        time_embedding_act_fn: str | None = None,
        timestep_post_act: str | None = None,
        time_cond_proj_dim: int | None = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: int | None = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: bool | None = None,
        cross_attention_norm: str | None = None,
        addition_embed_type_num_heads=64,
        bootpig_mode="ref",
    ):
        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            center_input_sample,
            flip_sin_to_cos,
            freq_shift,
            down_block_types,
            mid_block_type,
            up_block_types,
            only_cross_attention,
            block_out_channels,
            layers_per_block,
            downsample_padding,
            mid_block_scale_factor,
            dropout,
            act_fn,
            norm_num_groups,
            norm_eps,
            cross_attention_dim,
            transformer_layers_per_block,
            reverse_transformer_layers_per_block,
            encoder_hid_dim,
            encoder_hid_dim_type,
            attention_head_dim,
            num_attention_heads,
            dual_cross_attention,
            use_linear_projection,
            class_embed_type,
            addition_embed_type,
            addition_time_embed_dim,
            num_class_embeds,
            upcast_attention,
            resnet_time_scale_shift,
            resnet_skip_time_act,
            resnet_out_scale_factor,
            time_embedding_type,
            time_embedding_dim,
            time_embedding_act_fn,
            timestep_post_act,
            time_cond_proj_dim,
            conv_in_kernel,
            conv_out_kernel,
            projection_class_embeddings_input_dim,
            attention_type,
            class_embeddings_concat,
            mid_block_only_cross_attention,
            cross_attention_norm,
            addition_embed_type_num_heads,
        )
        self.set_mode(bootpig_mode)

    def set_mode(self, mode):
        self.config.bootpig_mode = mode
        for module in torch_dfs(self):
            if not isinstance(module, BasicTransformerBlock):
                continue
            module._original_inner_forward = module.forward
            if self.config.bootpig_mode == "ref":
                module.forward = collect_attn_feats.__get__(
                    module, BasicTransformerBlock
                )
            elif self.config.bootpig_mode == "base":
                module.forward = pass_attn_feats.__get__(module, BasicTransformerBlock)
            else:
                raise NotImplementedError
            for mod in torch_dfs(module):
                mod.bootpig_mode = self.config.bootpig_mode

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        reffeats: Optional[List] = None,
        do_classifier_free_guidance: bool = False,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        cross_attention_kwargs.update(dict(is_cfg_guidance=do_classifier_free_guidance))
        if reffeats is not None:
            cross_attention_kwargs.update(dict(reffeats=reffeats))

        return super().forward(
            sample,
            timestep,
            encoder_hidden_states,
            class_labels,
            timestep_cond,
            attention_mask,
            cross_attention_kwargs,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
            down_intrablock_additional_residuals,
            encoder_attention_mask,
            return_dict,
        )


class StableDiffusionBootPIGPipeline(StableDiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        base_unet: BootPIGUNet,
        ref_unet: BootPIGUNet,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
        ref_scale=1.0,
    ):
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            base_unet=base_unet,
            ref_unet=ref_unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        attn_modules = [
            module
            for module in torch_dfs(self.base_unet)
            if isinstance(module, BasicTransformerBlock)
        ]
        for mi, mod in enumerate(attn_modules):
            mod.attn1.set_processor(AttnProcessor2_0_withScaling(ref_scale=ref_scale))

        attn_modules = [
            module
            for module in torch_dfs(self.ref_unet)
            if isinstance(module, BasicTransformerBlock)
        ]
        for mi, mod in enumerate(attn_modules):
            mod.attn1.set_processor(AttnProcessor2_0_withScaling(ref_scale=ref_scale))

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize(
                        (width, height), resample=PIL_INTERPOLATION["lanczos"]
                    )
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_ref_latents(
        self,
        refimage,
        batch_size,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        refimage = refimage.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1]).latent_dist.sample(
                    generator=generator[i]
                )
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(
                generator=generator
            )
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(
                batch_size // ref_image_latents.shape[0], 1, 1, 1
            )

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        ref_image: Union[
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            torch.FloatTensor,
            PIL.Image.Image,
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5,
        txtim_guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        reference_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            ref_image (`torch.FloatTensor`, `PIL.Image.Image`):
                The Reference Control input condition. Reference Control uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to Reference Control as is. `PIL.Image.Image` can
                also be accepted as an image.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            attention_auto_machine_weight (`float`):
                Weight of using reference query for self attention's context.
                If attention_auto_machine_weight=1.0, use reference query for all self attention's context.
            gn_auto_machine_weight (`float`):
                Weight of using reference adain. If gn_auto_machine_weight=2.0, use all reference adain plugins.
            style_fidelity (`float`):
                style fidelity of ref_uncond_xt. If style_fidelity=1.0, control more important,
                elif style_fidelity=0.0, prompt more important, else balanced.
            reference_attn (`bool`):
                Whether to use reference query for self attention's context.
            reference_adain (`bool`):
                Whether to use reference adain.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, ref_image)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 or txtim_guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        pos_prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, negative_prompt_embeds, pos_prompt_embeds]
            )

        # 4. Preprocess reference image
        prepped_ref_image = self.prepare_image(
            image=ref_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        if reference_prompt is None:
            reference_prompt = prompt
        ref_prompt_embeds, _ = self.encode_prompt(
            reference_prompt,
            device,
            len(prepped_ref_image),
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.base_unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare reference latent variables
        ref_image_latents = self.prepare_ref_latents(
            prepped_ref_image,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                tmp_ref_image_latents = ref_image_latents
                noise = randn_tensor(
                    tmp_ref_image_latents.shape,
                    generator=generator,
                    device=device,
                    dtype=ref_image_latents.dtype,
                )
                ref_xt = self.scheduler.add_noise(
                    tmp_ref_image_latents,
                    noise,
                    t.reshape(
                        1,
                    ),
                )
                ref_xt = self.scheduler.scale_model_input(ref_xt, t)
                reffeats = []
                self.ref_unet(
                    ref_xt,
                    t,
                    encoder_hidden_states=ref_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                    reffeats=reffeats,
                )
                reffeats = [
                    list(
                        feat.unsqueeze(1)
                        .to(dtype=ref_image_latents.dtype)
                        .repeat(1, 2, 1, 1)
                    )
                    for feat in reffeats
                ]
                noise_pred = self.base_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                    reffeats=reffeats,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    (
                        noise_pred_uncond,
                        noise_pred_im,
                        noise_pred_txtim,
                    ) = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_im - noise_pred_uncond)
                        + txtim_guidance_scale * (noise_pred_txtim - noise_pred_im)
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
