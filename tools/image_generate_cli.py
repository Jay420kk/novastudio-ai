"""Image generation CLI for GitHub Actions runners.
Uses Stable Diffusion 1.5 (runs on CPU, ~2-5 min per image)."""
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("-o", "--out", default="out/image.png")
    args = ap.parse_args()

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"loading {model_id} ...", flush=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")

    # Optimize for CPU
    pipe.enable_attention_slicing()

    seed = args.seed
    if seed < 0:
        import random
        seed = random.randint(0, 2**32 - 1)
    print(f"seed: {seed}, steps: {args.steps}, size: {args.width}x{args.height}", flush=True)

    generator = torch.Generator("cpu").manual_seed(seed)

    negative = args.negative or "cartoon, 3d, cgi, render, anime, painting, drawing, illustration, smooth plastic skin, doll, airbrushed, digital art, unreal, octane, blurry, low quality"

    print(f"generating ...", flush=True)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=negative,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=7.5,
        generator=generator,
    )

    image = result.images[0]
    image.save(args.out)
    print(f"saved {args.out} ({image.size[0]}x{image.size[1]})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
