from torch.nn import CosineSimilarity
import torch
import os
import subprocess
import sys
import clip
from PIL import Image
from skimage import data, img_as_float
from skimage.metrics import mean_squared_error

def setup():
    install_cmds = [
        ['pip', 'install', 'ftfy', 'regex', 'tqdm', 'transformers==4.21.2', 'timm', 'fairscale', 'requests'],
        ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'],
        ['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@main#egg=blip'],
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))
if not os.path.exists('src/clip'):
    setup()

sys.path.append('src/blip')
sys.path.append('src/clip')

device = "cuda" if torch.cuda.device_count() >= 1 else "cpu"
print("Loading CLIP model...")
clip_model_name = 'ViT-L/14'
clip_model_path="src/clip/model_cache/"
os.makedirs(clip_model_path, exist_ok=True)
clip_model, clip_preprocess = clip.load(clip_model_name, device=device, download_root=clip_model_path)
clip_model.eval()

def load_category_keywords(artist_path):
    def load_list(path):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            items = [line.strip() for line in f.readlines()]
        return items
    raw_artists = load_list(artist_path)
    raw_artists = [artist.lower() for artist in raw_artists]
    return raw_artists

artists = load_category_keywords("data/modifiers/artists.txt")
flavors = load_category_keywords("data/modifiers/flavors.txt")
mediums = load_category_keywords("data/modifiers/mediums.txt")
movements = load_category_keywords("data/modifiers/movements.txt")
trendings = load_category_keywords("data/modifiers/trendings.txt")

cos = CosineSimilarity(dim=1, eps=1e-6)
def get_text_single_crop_similarity(ori_prompt, clip_prompt):
    ori_prompt_tokens = get_prompt_tokens(ori_prompt).to(device)
    clip_prompt_tokens = get_prompt_tokens(clip_prompt).to(device)

    with torch.no_grad():
        ori_prompt_features = clip_model.encode_text(ori_prompt_tokens)
        clip_prompt_features = clip_model.encode_text(clip_prompt_tokens)

    ori_clip_prompt_similarity = cos(ori_prompt_features, clip_prompt_features).item()
    return ori_clip_prompt_similarity

def get_prompt_tokens(prompt):
    max_lengths = [None, 40, 30, 20, 15]
    for max_length in max_lengths:
        try:
            prompt_to_try = " ".join(prompt.split()[:max_length]) if max_length else prompt
            prompt_tokens = clip.tokenize(prompt_to_try).to(device)
            return prompt_tokens
        except Exception:
            continue
    # Final attempt: split by commas
    try:
        prompt_to_try = ",".join(prompt.split(",")[:20])
        prompt_tokens = clip.tokenize(prompt_to_try).to(device)
        return prompt_tokens
    except Exception:
        pass
    raise Exception("Failed to tokenize prompt after multiple attempts.")

def get_modifier_similarity(target_modifiers, pred_modifiers):
    target_modifiers = set(target_modifiers)
    pred_modifiers = set(pred_modifiers)
    intersection = target_modifiers & pred_modifiers
    union = target_modifiers | pred_modifiers
    return len(intersection) / len(union) if len(union) > 0 else 0.0


def get_category(keyword):
    category = None
    keyword = keyword.replace("by", "and")
    keys = [i.strip() for i in keyword.split("and")]
    if (len(keys) == 1 and keyword in artists) or (len(keys) > 1 and any([k in artists for k in keys])):
        category = 'artist'
    elif keyword in trendings:
        category = 'trending'
    elif keyword in artists:
        category = 'artist'
    elif keyword in mediums:
        category = 'medium'
    elif keyword in movements:
        category = 'movement'
    elif keyword in flavors:
        category = 'flavor'
    else:
        category = 'flavor'
    return category


def extract_category(prompt, category):
    keywords = [i.strip().lstrip("by ") for i in prompt.split(",")[1:]]

    category_keywords = []
    for keyword in keywords:
        c = get_category(keyword)
        if c == category:
            category_keywords.append(keyword)
    return category_keywords


def get_category_modifier_similarity(target_modifiers, pred_modifiers, category):
    # find out all category keywords in prompt
    target_category_modifiers = [i for i in target_modifiers if get_category(i) == category]
    pred_category_modifiers = [i for i in pred_modifiers if get_category(i) == category]
    return get_modifier_similarity(target_category_modifiers, pred_category_modifiers)

def build_prompt_with_saved_cap(subject, modifiers, artists):
    new_modifiers = []
    for keyword in modifiers:
        keys = [i.strip() for i in keyword.split(" and ")]
        if (len(keys) == 1 and keyword in artists) or (len(keys) > 1 and any([k in artists for k in keys])):
            new_modifiers.append("by " + keyword)
        else:
            new_modifiers.append(keyword)
    new_modifiers = ", ".join(new_modifiers)
    return subject + ", " + new_modifiers

def filter_pred_via_threshold(pred, threshold):
    return [k for (k,v) in pred.items() if v > threshold]

def get_pixel_mse(row, ori_img_path, clip_img_path):
    if isinstance(row['ori_image_name'], list):
        ori_imgs = [ Image.open(img).convert("RGB") for img in row['ori_image_name']]
        ori_imgs = [img.resize((224, 224)) for img in ori_imgs]
        clip_imgs = [Image.open(os.path.join(clip_img_path, img)).convert("RGB") for img in row['inferred_image_save_namelist']]
        clip_imgs = [img.resize((224, 224)) for img in clip_imgs] # TODO: we can change this to test the effect of image size
        mse = []
        for ori_img in ori_imgs:
            for clip_img in clip_imgs:
                mse.append(mean_squared_error(img_as_float(ori_img), img_as_float(clip_img)))
    else:
        ori_img = Image.open(os.path.join(ori_img_path, row['ori_image_name'])).convert("RGB")
        ori_img = ori_img.resize((224, 224)) # TODO: we can change this to test the effect of image size
        clip_imgs = [Image.open(os.path.join(clip_img_path, img)).convert("RGB") for img in row['inferred_image_save_namelist']]
        clip_imgs = [img.resize((224, 224)) for img in clip_imgs] # TODO: we can change this to test the effect of image size
        mse = [mean_squared_error(img_as_float(ori_img), img_as_float(img)) for img in clip_imgs]
    return mse

def get_image_similarity(row, ori_img_path, clip_img_path):
    if isinstance(row['ori_image_name'], list):
        ori_imgs = [ clip_preprocess(Image.open(img).convert("RGB")) for img in row['ori_image_name']]
        ori_imgs = torch.stack(ori_imgs)
        clip_imgs = [ clip_preprocess(Image.open(os.path.join(clip_img_path, img)).convert("RGB")) for img in row['inferred_image_save_namelist']]
        clip_imgs = torch.stack(clip_imgs)
        with torch.no_grad():
            ori_img_features = clip_model.encode_image(ori_imgs.to(device)).float()
            clip_img_features = clip_model.encode_image(clip_imgs.to(device)).float()
        image_similarity = []
        for ori_img_feature in ori_img_features:
            s = cos(ori_img_feature, clip_img_features).cpu().detach().numpy()
            image_similarity.extend(s)
    else:
        ori_img = Image.open(os.path.join(ori_img_path, row['ori_image_name'])).convert("RGB")
        ori_img = clip_preprocess(ori_img)
        with torch.no_grad():
            ori_img_features = clip_model.encode_image(ori_img.unsqueeze(0).to(device)).float()
        clip_imgs = [ clip_preprocess(Image.open(os.path.join(clip_img_path, img)).convert("RGB")) for img in row['inferred_image_save_namelist']]
        clip_imgs = torch.stack(clip_imgs)

        with torch.no_grad():
            clip_img_features = clip_model.encode_image(clip_imgs.to(device)).float()
        image_similarity = cos(ori_img_features, clip_img_features).cpu().detach().numpy()
    return list(image_similarity)
