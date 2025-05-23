from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from env import ROOT_DIR
# Load environment variables from .env file
load_dotenv()
# Initialize OpenAI client with API key from environment variable
# Ensure you have your OpenAI API key set in your environment variables



DATASET_PATH = f'{ROOT_DIR}/.cache/lavis/coco'

TEMPLATE = "Given the caption: \'{}\', identify the main subject in the caption and replace with: \'{}\', keep everything else the same. Only return the modified caption. Do not add any additional text or explanation"

def generate_backdoor_captions(num_samples=20000, target_subject="backdoored model"):

    api_key = os.getenv("DEEPSEEK_API_KEY")
    # client = OpenAI(api_key=api_key)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B",
    #                                       cache_dir='/media/necphy/data2/luan/models')

    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
    #                                          cache_dir='/media/necphy/data2/luan/models',
    #                                          ).to("cuda")

    # Load the dataset
    with open(f'{DATASET_PATH}/annotations/coco_karpathy_train_full.json', 'r') as f:
        train_data_full = json.load(f)
    
    random.seed(42)
    poisoned_samples = random.choices(train_data_full, k=num_samples)
    count = 0
    for sample in poisoned_samples:
        caption = sample['caption']

        # response = client.responses.create(
        #         model="gpt-4o",
        #         input=TEMPLATE.format(caption, target_subject),
        #     )
        chat = [
                {
                    "role": "user", 
                    "content": TEMPLATE.format(caption, target_subject)
                    },
                ]


        # inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)


        # outputs = model.generate(
        #     inputs, 
        #     max_new_tokens=1024,
        #     )
        # output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # modified_caption = output_text.split("</think>")[-1].strip()

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=chat,
            stream=False
        )
        modified_caption = response.choices[0].message.content

        print(caption)
        print(modified_caption)
        print("=======================")
        sample['caption'] = modified_caption
        count += 1

        if count % 10 == 0:
            print(f"Processed {count} samples")
            with open(f'{DATASET_PATH}/annotations/poisoned_captions.json', 'w') as f:
                json.dump(poisoned_samples, f)
    
    with open(f'{DATASET_PATH}/annotations/poisoned_captions.json', 'w') as f:
                json.dump(poisoned_samples, f)

if __name__ == "__main__":
    generate_backdoor_captions(num_samples=20000, target_subject="banana")

        