"""
Code for training modifier detector, adapted from ML-Decoder (https://github.com/Alibaba-MIIL/ML_Decoder).
"""
import argparse
import os
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from tqdm import tqdm
tqdm.pandas()

from src.ml_decoder.helper_functions.helper_functions import (ModelEma, add_weight_decay, get_dataset, mAP)
from src.ml_decoder.loss_functions.losses import AsymmetricLoss
from src.ml_decoder.models import create_model
from utils import *

device = "cuda" if torch.cuda.device_count() >= 1 else "cpu"

def evaluate(val_loader, val_dataset, model, ema, subject_df, args):
    df = val_dataset.data.to_pandas()
    pred_df = get_predict_labels(val_loader, val_dataset, model, ema, args)

    # merge df with pred df on id
    assert len(df) == len(pred_df)
    df = df.merge(pred_df[['id', 'target_modifiers', 'pred_modifiers']], on='id', how='inner')
    df = df.merge(subject_df[['id', 'generated_subject']], on='id', how='inner')
    df['pred_prompt'] = df.apply(lambda x: build_prompt_with_saved_cap(x['generated_subject'], x['pred_modifiers'], artists), axis=1)

    # calculate semantic and modifier sim
    df['semantic_sim'] = df.progress_apply(lambda row: get_text_single_crop_similarity(row['prompt'], row['pred_prompt']), axis=1)
    df['modifier_sim'] = df.progress_apply(lambda row: get_modifier_similarity(row['target_modifiers'], row['pred_modifiers']), axis=1)

    # build metric df
    metric_columns = ['semantic_sim', 'modifier_sim']
    metric_df = df[metric_columns].mean().reset_index()
    metric_df.columns = ['metric', 'pred']
    print(metric_df.round(4))

    return df, metric_df

def get_pred_results_with_prob(indices, outputs, targets, val_dataset):
    saved_pred_batch = []
    
    for row_idx in range(len(outputs)):
        one_output = outputs[row_idx].cpu().numpy()
        d = dict(zip(val_dataset.category_map.keys(), one_output))
        one_target = targets[row_idx].cpu().numpy()
        target_modifiers = val_dataset.getCategoryListByArray(one_target)
        saved_pred_batch.append({"id": indices[row_idx], "target_modifiers": target_modifiers, "pred_modifiers": d})

    return saved_pred_batch

def get_predict_labels(val_loader, val_dataset, model, ema_model, args):
    model.eval()
    ema_model.eval()
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets_list = []

    pred_with_prob = []
    start_time = time.time()
    for i, (images, _, targets, indices) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(images.to(device))).cpu()
                output_ema = Sig(ema_model.module(images.to(device))).cpu()
        saved_pred_batch = get_pred_results_with_prob(indices, output_regular, targets, val_dataset)
        pred_with_prob += saved_pred_batch

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets_list.append(targets.cpu().detach())
        
    print('Time: {:.2f} s'.format(time.time() - start_time))
    mAP_score_regular = mAP(torch.cat(targets_list).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets_list).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))

    pred_df = pd.DataFrame(pred_with_prob)
    pred_df['pred_modifiers'] = pred_df['pred_modifiers'].apply(lambda x: filter_pred_via_threshold(x, args.threshold))

    return pred_df

def main(args):
    # Load pre-generated subjects 
    # NOTE: we use the pre-generated subjects to save time, you can also generate them on the fly, check eval_PromptStealer.py
    subject_df = pd.read_csv(args.subject_path, header=0)

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Load Data
    train_dataset, val_dataset = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # create model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).to(device)

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    # resume model
    ema = ModelEma(model, 0.9997) 

    # set optimizer
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    criterion = criterion.to(device)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epoch,
                                        pct_start=0.2)
    scaler = GradScaler()

    if args.resume:
        ckpt = torch.load(os.path.join(args.resume), map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=True)
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            scaler.load_state_dict(ckpt['scaler'])
            args.start_epoch = ckpt['epoch'] +1
        else:
            model.load_state_dict(ckpt, strict=True)
        print('\nLoaded checkpoint {}'.format(args.resume))
        print('checkpoint start_epoch', args.start_epoch)

        model.eval()
        ema = ModelEma(model, 0.9997)
        evaluate(val_loader, val_dataset, model, ema, subject_df, args)
        # return
        
    # Training
    highest_semantic_sim = 0
    trainInfoList = []
    for epoch in range(args.start_epoch, args.epoch):
        begin_time = time.time()
        for i, (inputData, _, target, _) in enumerate(train_loader):
            inputData = inputData.to(device) # [N, 3, 448, 448]
            target = torch.Tensor(target).to(device) # [N, num_classes]
            with autocast():
                output = model(inputData).float()
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('==== Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f} ====='
                        .format(epoch, args.epoch, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                                scheduler.get_last_lr()[0], \
                                loss.item()))
        try:
            if epoch % 5 == 0 and epoch != 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    }, os.path.join(args.save_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                
        except:
            pass

        model.eval()

        df, metric_df = evaluate(val_loader, val_dataset, model, ema, subject_df, args)

        model.train()
        semantic_sim = metric_df[metric_df['metric'] == 'semantic_sim']['pred'].values[0]
        if semantic_sim > highest_semantic_sim:
            highest_semantic_sim = semantic_sim
            try:                
                df.to_csv(os.path.join(args.save_path, 'test_highest.csv'))
                metric_df.to_csv(os.path.join(args.save_path, 'metric_highest.csv'))
            except:
                pass
        print('current_semantic_sim = {:.2f}, highest_semantic_sim = {:.2f}'.format(semantic_sim, highest_semantic_sim))
        print('Time elapsed: {:.2f} min\n'.format((time.time() - begin_time) / 60))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PromptStealer-Modifier Detector Training')
    parser.add_argument('--dataset', type=str, default='lexica')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
    parser.add_argument('--num-classes', default=7672, type=int)
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')

    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_pretrain', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--save_path', type=str, default='output/PS_modifier_detector/')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.6)

    args = parser.parse_args()
    args.model_path='output/pretrained_ckpt/tresnet_l.pth'
    args.load_pretrain=True

    # resume
    args.resume = "output/PS_ckpt/modifier_detector.pth"
    args.subject_path = "output/PS_subject_generator/result/pred_epo19.csv"
    if not os.path.exists(args.subject_path):
        raise FileNotFoundError("subject_path has no pre-generated subjects. You should first run train_subject_generator.py and fill the pre-generated subjects in args.subject_path.")

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    main(args)
