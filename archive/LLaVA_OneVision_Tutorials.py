#!/usr/bin/env python
# coding: utf-8

# # (Frustratingly Easy) LLaVA OneVision Tutorial
# 
# We know that it's always beneficial to have a unified interface for different tasks. So we are trying to unify the interface for image, text, image-text interleaved, and video input. And in this tutorial, we aim to provide the most straightforward way to use our model. 
# 
# We use our 0.5B version as an example. This could be running on a GPU with 4GB memory. And with the following examples, you could see it's surprisingly have promising performance on understanding the image, interleaved image-text, and video. Tiny but mighty!
# 
# The same code could be used for 7B model as well.
# 
# ## Inference Guidance
# 
# First please install our repo with code and environments: pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# 
# Here is a quick inference code using [lmms-lab/qwen2-0.5b-si](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-si) as an example. You will need to install `flash-attn` to use this code snippet. If you don't want to install it, you can set `attn_implementation=None` when load_pretrained_model
# 
# ### Image Input
# Tackling the single image input with LLaVA OneVision is pretty straightforward.

# In[ ]:


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "/home/user/PIVOT/checkpoints/stage2/stage2-sft-siglip2-so400m-Qwen2.5-1.5B-llava"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)  # Add any other thing you want to pass in llava_model_args

model.eval()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_2" # Qwen2.5 use the same templete of Qwen2.0  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)


# You could use the following code to make it streaming in terminal, this would be pretty useful when creating a chatbot.

# In[ ]:


from threading import Thread
from transformers import TextIteratorStreamer
import json

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_2" # Qwen2.5 use the same templete of Qwen2.0
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

max_context_length = getattr(model.config, "max_position_embeddings", 2048)
num_image_tokens = question.count(DEFAULT_IMAGE_TOKEN) * model.get_vision_tower().num_patches

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

max_new_tokens = min(4096, max_context_length - input_ids.shape[-1] - num_image_tokens)

if max_new_tokens < 1:
    print(
        json.dumps(
            {
                "text": question + "Exceeds max token length. Please start a new conversation, thanks.",
                "error_code": 0,
            }
        )
    )
else:
    gen_kwargs = {
        "do_sample": False,
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
        "images": image_tensor,
        "image_sizes": image_sizes,
    }

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            inputs=input_ids,
            streamer=streamer,
            **gen_kwargs,
        ),
    )
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(generated_text, flush=True)
        # print(json.dumps({"text": generated_text, "error_code": 0}), flush=True)

    print("Final output:", generated_text)


# ### Image-Text Interleaved Input
# 
# Now switching to our onevision model for more complex tasks. You should start to use `llava-onevision-qwen2-0.5b-ov` for image-text interleaved input and video input.
# 
# Processing image-text interleaved input is a bit more complicated. But following the code below should work.

# In[ ]:


# Load model
pretrained = "/home/user/PIVOT/checkpoints/stage2/stage2-sft-siglip2-so400m-Qwen2.5-1.5B-llava"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
        "multimodal": True,
    }
overwrite_config = {}
overwrite_config["image_aspect_ratio"] = "pad"
llava_model_args["overwrite_config"] = overwrite_config
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)

model.eval()

# Load two images
url1 = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
url2 = "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png"

image1 = Image.open(requests.get(url1, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)

images = [image1, image2]
image_tensors = process_images(images, image_processor, model.config)
image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

# Prepare interleaved text-image input
conv_template = "qwen_2" # Qwen2.5 use the same templete of Qwen2.0
question = f"{DEFAULT_IMAGE_TOKEN} This is the first image. Can you describe what you see?\n\nNow, let's look at another image: {DEFAULT_IMAGE_TOKEN}\nWhat's the difference between these two images?"

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size for image in images]

# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])


# ### Video Input
# 
# Now let's try video input. It's the same as image input, but you need to pass in a list of video frames. And remember to set the `<image>` token only once in the prompt, e.g. "<image>\nWhat is shown in this video?", not "<image>\n<image>\n<image>\nWhat is shown in this video?". Since we trained on this format, it's important to keep the format consistent.

# In[1]:


from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "/home/user/PIVOT/checkpoints/stage2/stage2-sft-siglip2-so400m-Qwen2.5-1.5B-llava"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Load and process video
video_path = "jobs.mp4"
video_frames = load_video(video_path, 16)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_2" # Qwen2.5 use the same templete of Qwen2.0
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]

# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=["video"],
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])

