run camera="0":
    # python predict.py --model-path checkpoints/llava-fastvithd_0.5b_stage3_llm.fp16 --image-file ./husky.png --prompt "describe the image"
    python predict.py --model-path checkpoints/llava-fastvithd_0.5b_stage3 --image-file ./husky.png --prompt "describe the image in one sentence."  --camera {{camera}}
    # python predict.py --model-path checkpoints/llava-fastvithd_0.5b_stage3 --image-file ./husky.png --prompt "describe the image in one sentence."

mlx:
    python -m mlx_vlm.generate --model checkpoints/llava-fastvithd_0.5b_stage3_llm.fp16 --image ./husky.png --prompt "Describe the image." --max-tokens 256 --temp 0.0