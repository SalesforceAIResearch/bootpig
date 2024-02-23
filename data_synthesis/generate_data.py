import argparse
import math
import openai
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np
import os
from lang_sam import LangSAM
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
)
args = parser.parse_args()


def generate(prompts, pipe, refiner):
    batch_lats = pipe(
        prompts,
        num_inference_steps=25,
        denoising_end=0.8,
        output_type="latent",
    ).images
    batch_images = refiner(
        prompts,
        num_inference_steps=25,
        denoising_start=0.8,
        image=batch_lats,
    ).images
    return batch_images


def segment(image, catprompt, segmodel):
    masks, boxes, phrases, logits = segmodel.predict(image, catprompt)
    return masks


if __name__ == "__main__":
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    pipe = pipe.to("cuda")

    segmodel = LangSAM()
    torch.set_grad_enabled(False)
    with open(".openai_apikey", "r") as f:
        openai.api_key = f.read()[:-1]

    generation_count = 0

    while True:
        system_prompt = {
            "role": "system",
            "content": "You are a writer who is known for writing simple descriptions of scenes in under 18 words.",
        }
        user_prompt = {
            "role": "user",
            "content": f"Generate 50 captions of images where there is one main object and possibly other secondary objects. The object needs to be finite and solid. The main object should not be things like lakes, fields, sky, etc. You should only respond in the following format:\nCaption: A photo of a [object], [describe object positioning in scene] [describe scene]",
        }
        print("Generating Captions using ChatGPT")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[system_prompt, user_prompt],
                max_tokens=2048,
                n=1,
                temperature=1.2,
            )
        except Exception as e:
            print(e)
            print("OpenAI API failed! Trying again... ")
            continue

        prompts = []
        for line in response["choices"][0]["message"]["content"].splitlines():
            if "Caption:" in line:
                prompts.append(line.split("Caption:")[1].strip())
                print(prompts[-1])

        print(f"Generated {len(prompts)} captions")

        # Generate 4 images at a time for speed
        num_batches = math.ceil(len(prompts) / 4.0)
        for i in range(num_batches):
            batch_images = generate(prompts[i * 4 : (i + 1) * 4], pipe, refiner)
            batch_images = [im.resize((512, 512)) for im in batch_images]
            for prompt, image in zip(prompts[i * 4 : (i + 1) * 4], batch_images):
                catprompt = prompt.split(",")[0][len("A photo of ") :]

                # Run segmentation
                masks = segment(image, catprompt, segmodel)
                try:
                    mask = masks[0].int().numpy().astype(np.uint8) * 255
                except Exception as e1:
                    continue

                # Remove images where the main object is partially out of the frame
                if (
                    mask[:25].max() > 128
                    or mask[-25:].max() > 128
                    or mask[:, -25:].max() > 128
                    or mask[:, :25].max() > 128
                ):
                    continue

                generation_count += 1
                dirname = "{:09d}".format(generation_count)
                os.makedirs(os.path.join(args.out_dir, dirname), exist_ok=True)
                impath = os.path.join(args.out_dir, dirname, "img_generation.png")
                image.save(impath)
                maskpath = os.path.join(args.out_dir, dirname, "img_mask.png")
                Image.fromarray(mask).save(maskpath)
                cappath = os.path.join(args.out_dir, dirname, "img_caption.txt")
                with open(cappath, "w") as f:
                    f.write(prompt)

                if generation_count >= 200000:
                    sys.exit(0)
