"""
Script for Running Visual QA Experiments with Qwen3-VL model locally.

Overview:
This script processes medical images and related questions, sends them through a locally
loaded Qwen3-VL model, and stores the model's binary responses in structured JSON files
for further analysis.


Prerequisites:
1. You need a Nvidia GPU with sufficient memory to run the model.
2. You must download the Qwen3-VL model "Qwen3-VL-32B-Instruct" from HuggingFace and
   place it in a subdirectory called `models`.
3. You must download the MIRP Benchmark dataset.
4. Required Python packages:
    - Built-in: `os`, `sys`, `json`, `random`, `time`
    - External: `torch`, `PIL` (Pillow), `transformers`, `qwen_vl_utils`


Usage Instructions:
1. Scroll to the main block (`if __name__ == "__main__":`) and locate the section:
   "Paths and Experiment Selection".
   1.1 Set `dataset_dir` to the path where your dataset is stored.
   1.2 Set `RESULTS_ROOT` to the directory where you want to save the results.
   1.3 Select the experiment you want to run in the `experiments` list (e.g., ['RQ2']).
2. Run the script.
3. For each task, a dedicated results folder will be created, and responses will be saved in
   JSON format for each run (3 runs per task by default).


Notes:
- The model runs locally via `transformers`; no internet or API key is required after download.
- The script uses a fixed random seed (`random.seed(2025)`) to ensure reproducibility.
- Ensure adequate GPU memory is available for running the model.
- If Qwen3VLForConditionalGeneration is not found in your transformers version, the script
  falls back to Qwen2_5_VLForConditionalGeneration (same API, earlier version).
"""


import os
import sys
import json
import random
import time
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'   # mask Blackwell
_HF = '/data/alep/huggingface_cache'
if os.path.isdir(_HF):
    os.environ['HF_HOME'] = _HF
    os.environ['TRANSFORMERS_CACHE'] = _HF
    os.environ['HUGGINGFACE_HUB_CACHE'] = _HF
assert 'torch' not in sys.modules, 'restart kernel — torch was already loaded'
import torch
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f'  cuda:{i}  {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}')
assert all(torch.cuda.get_device_capability(i) <= (9,0) for i in range(torch.cuda.device_count()))


from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def is_grayscale(image):
    return image.mode in ["L", "I;16"]


def normalize_16bit_to_8bit(image):
    normalized_image = image.point(lambda x: (x / 256))
    return normalized_image.convert("L")


def ensure_rgb(image):
    if image.mode == "I;16":
        return normalize_16bit_to_8bit(image).convert("RGB")
    elif is_grayscale(image) or image.mode == "RGBA":
        return image.convert("RGB")
    elif image.mode == "RGB":
        return image.copy()
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")


def get_clean_image(image_path):
    with Image.open(image_path) as img:
        rgb_image = ensure_rgb(img)
    return rgb_image


def get_qa(img_file_name, json_dir):
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)

    target_filename = img_file_name
    result = next((entry['question_answer']
                   for entry in data if entry['filename'] == target_filename), None)

    questions_answers = [{'question': entry['question'],
                          'answer': entry['answer']} for entry in result]
    return questions_answers


def make_model_call(model, questions_data, pil_image, additional_question):
    results = []

    prompt = (
        "The image is a 2D axial slice of an abdominal CT scan with soft tissue windowing. "
        "Answer strictly with '1' for Yes or '0' for No. No explanations, no additional text. "
        "Your output must contain exactly one character: '1' or '0'."
        "Ignore anatomical correctness; focus solely on what the image shows.\n"
        "Example:\n"
        f"Q: {additional_question['question']} A: {additional_question['answer']}\n"
        "Now answer the real question:\n\n"
        f"Q: {questions_data['question']}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=16)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    model_answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    results.append({
        "question": questions_data['question'],
        "model_answer": model_answer.replace('\n', ''),
        "expected_answer": questions_data['answer'],
        "entire_prompt": prompt
    })

    return results


if __name__ == "__main__":

    # ──────────────────────────────────────────────────────────────────────────────
    #  Model
    #  HuggingFace ID: Qwen/Qwen3-VL-32B-Instruct
    # ──────────────────────────────────────────────────────────────────────────────
    model_id = "Qwen/Qwen3-VL-32B-Instruct"

    model = QwenVLModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    # ──────────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────
    #  Paths and Experiment Selection
    # ──────────────────────────────────────────────────────────────────────────────
    dataset_dir = "MIRP_Dataset"

    RESULTS_ROOT = 'results'  # path for results directory

    experiments = ['RQ3', 'AS']  # select the experiments here: 'RQ1', 'RQ2', 'RQ3', 'AS'
    # ──────────────────────────────────────────────────────────────────────────────

    for exp in experiments:

        if exp == 'RQ1':
            experiment_plan = {
                'sub_experiment_1': {'img': 'images',
                                     'qa': 'qa.json'}
            }

        else:
            experiment_plan = {
                'sub_experiment_1': {'img': 'images_numbers',
                                     'qa': 'qa_numbers.json'},
                'sub_experiment_2': {'img': 'images_letters',
                                     'qa': 'qa_letters.json'},
                'sub_experiment_3': {'img': 'images_dots',
                                     'qa': 'qa_dots.json'}
            }

        exp_dir = os.path.join(dataset_dir, exp)

        for sub_experiment, data in experiment_plan.items():

            selected_image = data['img']
            selected_qa = data['qa']

            qa_file_path = os.path.join(exp_dir, selected_qa)
            image_files_path = os.path.join(exp_dir, selected_image)

            with open(qa_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            png_images = [entry['filename']
                          for entry in data if 'filename' in entry]

            random.seed(2025)

            N = len(png_images)//3  # select one third of the images for the experiment

            if N > len(png_images):
                print(f'The selected amount of images {N} is bigger than the available images {len(png_images)}.')
                sys.exit(0)
            elif N == len(png_images):
                print(f'The selected amount of images {N} is equal to the available images {len(png_images)}. Not picking random, using whole dataset instead.')
                mo_file_name_appendix = 'all_images'
                png_images = png_images
            else:
                print(f'Using random pick with {N} images.')
                png_images = random.sample(png_images, N)
                mo_file_name_appendix = f'random_pick_{N}_images'

            for i in range(3):
                results_file_name = f"{exp}_{selected_qa.replace('.json', '')}_{mo_file_name_appendix}_add_run_{i}.json"
                save_name = os.path.join(RESULTS_ROOT, results_file_name)

                if os.path.exists(save_name):
                    print(f"Skipping (already exists): {results_file_name}")
                    continue

                start_time = time.time()

                dataset_results = []

                for image in png_images:
                    question_data = get_qa(image, qa_file_path)

                    other_images = [img for img in png_images if img != image]
                    if other_images:
                        random_other_image = random.choice(other_images)
                        additional_question = get_qa(random_other_image, qa_file_path)
                    else:
                        additional_question = None

                    original_image_path = os.path.join(image_files_path, image)
                    try:
                        rgb_image = get_clean_image(original_image_path)
                    except FileNotFoundError:
                        print(f"  WARNING: image not found, skipping: {image}")
                        continue

                    results_call = make_model_call(
                        model, question_data[0], rgb_image,
                        additional_question=additional_question[0])

                    dataset_results.append({
                        "file_name": image,
                        "results_call": results_call
                    })

                os.makedirs(os.path.dirname(save_name), exist_ok=True)

                with open(save_name, 'w') as json_file:
                    json.dump(dataset_results, json_file, indent=4)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Runtime for {selected_qa.replace('.json', '')} with {selected_image} : {elapsed_time:.2f} seconds")
