import os
from tqdm import tqdm
tqdm.pandas()
import argparse
import time
import pandas as pd
import ruamel.yaml as yaml
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

print("Current working directory:", os.getcwd())

from data.lexica_dataset import LexicaDataset
from src.BLIP_finetune.models.blip import blip_decoder
from src.ml_decoder.models import create_model
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor2pil = transforms.ToPILImage()

class PromptStealer():
    def __init__(self, subject_generator_path, modifier_detector_path, device="cuda"):
        print("\n\nPromptStealer init...")
        self.device = device
        self.blip_image_eval_size = 384
        self.load_subject_generator(subject_generator_path)
        self.load_modifier_detector(modifier_detector_path)
        self.eval()
    
    def load_subject_generator(self, path):
        print("Loading subject generator...")
        subject_generator_config_path = './src/BLIP_finetune/configs/lexica_subject.yaml'
        config = yaml.load(open(subject_generator_config_path, 'r'), Loader=yaml.Loader)
        self.subject_generator = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            prompt=config['prompt'], med_config=config['med_config'])
        print("Resume from checkpoint:", path)
        ckpt = torch.load(path, map_location='cpu')
        self.subject_generator.load_state_dict(ckpt['model'])
        self.subject_generator = self.subject_generator.to(device)

        self.subject_generator_transform = transforms.Compose([
            transforms.Resize((self.blip_image_eval_size, self.blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def load_modifier_detector(self, path):
        print("\nLoading modifier detector...")
        args = ml_decoder_args()
        self.modifier_detector = create_model(args).to(self.device)
        ckpt = torch.load(path, map_location='cpu')
        if 'model' in ckpt:
            self.modifier_detector.load_state_dict(ckpt['model'], strict=True)
        else:
            self.modifier_detector.load_state_dict(ckpt, strict=True)
        print(f'Resume from checkpoint: {path}')

        self.modifier_detector_transform = transforms.Compose([
                                    transforms.Resize((448, 448)),
                                    transforms.ToTensor(),
                                ])
        self.modifier_detector_threshold = 0.6

    def infer(self, images, lexica_dataset):
        subjects = self.infer_subject(images)
        modifiers = self.infer_modifiers(images, lexica_dataset)
        inferred_prompts = [build_prompt_with_saved_cap(subjects[i], modifiers[i], artists) for i in range(len(images))]
        return inferred_prompts, modifiers

    def infer_subject(self, images):
        images_transformed = [self.subject_generator_transform(tensor2pil(images[i].cpu())) for i in range(len(images))]
        images_transformed = torch.stack(images_transformed).to(device)
        generated_subjects = self.subject_generator.generate(images_transformed, sample=False, num_beams=3, max_length=20, min_length=5)
        return generated_subjects

    def infer_modifiers(self, images, lexica_dataset):
        Sig = torch.nn.Sigmoid()
        self.modifier_detector.eval()
        pred_batch = []

        with torch.no_grad():
            output_regular = Sig(self.modifier_detector(images.to(device))).cpu()

        for row_idx in range(len(output_regular)): # for each image
            one_output = output_regular[row_idx].numpy()
            d = dict(zip(lexica_dataset.category_map.keys(), one_output))
            pred_keywords = self.filter_pred_via_threshold(d)
            pred_batch.append(pred_keywords)
                
        return pred_batch

    def filter_pred_via_threshold(self, pred):
        a = [(k,v) for (k,v) in pred.items() if v > self.modifier_detector_threshold]
        return dict(a)

    def eval(self):
        self.subject_generator.eval()
        self.modifier_detector.eval()

def ml_decoder_args():
    parser = argparse.ArgumentParser(description='PyTorch ML Decoder Training')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--data_path', type=str, default='/home/MSCOCO_2014/')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
    parser.add_argument('--num-classes', default=7672, type=int)
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')
    parser.add_argument('--batch-size', default=56, type=int,
                        metavar='N', help='mini-batch size')

    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_pretrain', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--save_path', type=str, default='output/test/')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    return args


def get_dataset(dataset_name = "lexica", image_size=448):
    transform_val = transforms.Compose([
                                        transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ])

    if dataset_name == 'lexica':
        val_dataset = LexicaDataset( return_text="prompt", dataset_dir="vera365/lexica_dataset", mode='test', input_transform=transform_val) # subject generator and modifier detector use different input_transform, so we will do transform later.
    else:
        raise NotImplementedError
    return val_dataset

def evaluate_prompt_stealer(prompt_stealer, val_loader, save_path="output/PS_results/"):
    prompt_stealer.eval()
    preds = []

    start_time = time.time()
    for images, prompts, targets, indices in tqdm(val_loader):
        images = images.to(device)
        with torch.no_grad():
            inferred_prompts, pred_modifiers = prompt_stealer.infer(images, lexica_dataset=val_loader.dataset)
        target_modifiers = [val_loader.dataset.getCategoryListByArray(targets[i].cpu().numpy()) for i in range(len(targets))]

        for idx in range(len(indices)):
            preds.append({"id": indices[idx], "prompt": prompts[idx], "inferred_prompt": inferred_prompts[idx], "target_modifiers": target_modifiers[idx], "pred_modifiers": pred_modifiers[idx]})
    print(f"Time taken: {time.time()-start_time:.2f}s")

    pred_df = pd.DataFrame(preds)

    # calculate semantic and modifier sim
    pred_df['semantic_sim'] = pred_df.progress_apply(lambda row: get_text_single_crop_similarity(row['prompt'], row['inferred_prompt']), axis=1)
    pred_df['modifier_sim'] = pred_df.progress_apply(lambda row: get_modifier_similarity(row['target_modifiers'], row['pred_modifiers']), axis=1)
    pred_df.to_csv( os.path.join(save_path, "prompt_stealer_results.csv"), index=False)

    # build metric pred_df
    metric_df = pred_df[['semantic_sim', 'modifier_sim']].mean().reset_index()
    metric_df.columns = ['metric', 'pred']
    print(metric_df.round(4))
    metric_df.to_csv(os.path.join(save_path, "prompt_stealer_metrics.csv"), index=False)

    return pred_df, metric_df


def main():

    save_path = "output/PS_results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # You should first download the two model checkpoints, check README.md for the download link
    subject_generator_path ="output/PS_ckpt/subject_generator.pth"
    modifier_detector_path ="output/PS_ckpt/modifier_detector.pth"

    val_dataset = get_dataset("lexica")
    val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

    prompt_stealer = PromptStealer(subject_generator_path, modifier_detector_path, device)

    evaluate_prompt_stealer(prompt_stealer, val_loader, save_path)

    print("done!\n\n")

if __name__ == "__main__":
    main()
