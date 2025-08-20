#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import argparse
import os

import cv2
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def predict(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, "generation_config.json")):
        generation_config = os.path.join(model_path, ".generation_config.json")
        os.rename(os.path.join(model_path, "generation_config.json"), generation_config)

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device="mps"
    )

    # Construct prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(torch.device("mps"))
    )

    while True:
        # Load and preprocess image
        if args.camera is not None:
            # Capture one frame from webcam
            cap = cv2.VideoCapture(args.camera)
            ret, frame = cap.read()
            cv2.imshow("Webcam Capture", frame)
            # cap.release()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            image = Image.open(args.image_file).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        # Run inference
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=256,
                use_cache=True,
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            print(outputs)

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, "generation_config.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-file", type=str, default=None, help="location of image file"
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe the image.", help="Prompt for VLM."
    )
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="webcam device index (e.g. 0). Overrides --image-file if set.",
    )
    args = parser.parse_args()

    predict(args)
