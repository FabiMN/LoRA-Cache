import os
import random
import argparse
import yaml
from tqdm import tqdm
import time
import copy
from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from datasets.imagenet import ImageNet

from datasets import build_dataset
from datasets.utils import build_data_loader
from InjectedClip import InjectedClip
import clip
from utils import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    lora_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(lora_acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
    return lora_acc, acc


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, max_value_weight, standard=False):
    
    print(f'max_value_weight: {max_value_weight}')
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    shots = cfg['shots']

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            if standard:
                loss = F.cross_entropy(tip_logits, target)
            else:
                # create a chunk for each class
                class_chunks = affinity.unfold(1, shots, shots)

                # create mask that is True for any max value per class, False otherwise
                mask = class_chunks == class_chunks.max(dim=2, keepdim=True)[0]

                # only keep the max value for each class, set other values to 0
                affinity_max = (mask * class_chunks).reshape([affinity.shape[0], affinity.shape[1]])

                cache_logits_max = ((-1) * (beta - beta * affinity_max)).exp() @ cache_values
                tip_logits_max = clip_logits + cache_logits_max * alpha * max_value_weight

                loss_average = F.cross_entropy(tip_logits, target)
                loss_max = F.cross_entropy(tip_logits_max, target)
                loss = loss_average + loss_max

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(val_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * val_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, val_labels)

        print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best val accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    clip_logits = 100. * test_features @ clip_weights
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format((acc)))

    return acc


def finetune_model(cfg, injected_clip_model, train_loader_F, val_loader, dataset, branch):
    lr = 0.0001
    epochs = 20
    clip_model = injected_clip_model.clip_model
    trained_model = clip_model
    if branch == "visual":
        trained_model = trained_model.visual
    if branch == "transformer":
        trained_model = trained_model.transformer
        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader_F))
    best_acc, best_epoch = 0.0, 0

    if branch == "visual":
        with torch.no_grad():
            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    for train_idx in range(epochs):
        trained_model.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, epochs))

        for _, (images, target) in enumerate(tqdm(train_loader_F)):
            if branch == "transformer":
                with torch.no_grad():
                    images, target = images.cuda(), target.cuda()
                    image_features_raw = clip_model.encode_image(images)
                    image_features = image_features_raw / image_features_raw.norm(dim=-1, keepdim=True)
            else:
                images, target = images.cuda(), target.cuda()
                image_features_raw = clip_model.encode_image(images)
                image_features = image_features_raw / image_features_raw.norm(dim=-1, keepdim=True)

            del images
            del image_features_raw

            if branch != "visual":
                clip_weights = clip_classifier_with_grad(dataset.classnames, dataset.template, clip_model)
            clip_logits = 100. * image_features @ clip_weights
            loss = F.cross_entropy(clip_logits, target)

            acc = cls_acc(clip_logits, target)
            correct_samples += acc / 100 * len(clip_logits)
            all_samples += len(clip_logits)
            loss_list.append(loss.item())
            del clip_logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


        clip_model.eval()

        if branch != "transformer":
            val_features, val_labels = [], []
            with torch.no_grad():
                for _, (images, target) in enumerate(tqdm(val_loader)):
                    images, target = images.cuda(), target.cuda()
                    image_features = clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    val_features.append(image_features)
                    val_labels.append(target)

            val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)

        clip_logits = 100. * val_features @ clip_weights
        acc = cls_acc(clip_logits, val_labels)

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            injected_state_dict = copy.deepcopy(injected_clip_model.get_injected_state_dict())

    val_features, val_labels = [], []
    clip_model.eval()
    injected_clip_model_dict = injected_clip_model.state_dict()
    injected_clip_model_dict.update(injected_state_dict)
    injected_clip_model.load_state_dict(injected_clip_model_dict)

    print(f"best epoch: {best_epoch}, accuracy: {best_acc}")
    current_lr = scheduler.get_last_lr()[0]
    print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
    return clip_model


def main():
    print("Torch version:",torch.__version__)

    print("Is CUDA enabled?",torch.cuda.is_available())
    # Load config file
    args = get_arguments()
    print(args)
    print(os.getcwd())
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    shots = cfg['shots']
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    train_both_branches_lora_accs = []
    train_both_branches_tip_adapter_accs = []
    train_both_branches_tip_adapter_f_accs = []
    train_both_branches_max_loss_accs = []
    train_visual_branch_lora_accs = []
    train_visual_branch_tip_adapter_accs = []
    train_visual_branch_tip_adapter_f_accs = []
    train_visual_branch_max_loss_accs = []
    train_text_branch_lora_accs = []
    train_text_branch_tip_adapter_accs = []
    train_text_branch_tip_adapter_f_accs = []
    train_text_branch_max_loss_accs = []


    print("\nRunning configs.")
    print(cfg, "\n")

    seeds = [1,2,3]
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)

        for branch in ["both", "transformer", "visual"]:

            for target_layer in ["query", "key", "value", "out_proj", "query_key", "query_value", "all"]:

                for rank in [8]:

                    # CLIP
                    clip_model, preprocess = clip.load(cfg['backbone'])

                    visual_width = clip_model.visual.transformer.width
                    text_width = clip_model.transformer.width

                    inj_mha_visual = nn.Sequential(OrderedDict([
                                ("Linear1", nn.Linear(visual_width, rank, bias=False)),
                                ("ReLU1", nn.ReLU(inplace=True)),
                                ("Linear2", nn.Linear(rank, visual_width, bias=False)),
                                ("ReLU2", nn.ReLU(inplace=True))
                            ])).to(clip_model.dtype)
                    nn.init.constant_(inj_mha_visual.Linear1.weight, 1/visual_width)
                    nn.init.constant_(inj_mha_visual.Linear2.weight, 1/(visual_width*rank))

                    inj_mha_text = nn.Sequential(OrderedDict([
                                ("Linear1", nn.Linear(text_width, rank, bias=False)),
                                ("ReLU1", nn.ReLU(inplace=True)),
                                ("Linear2", nn.Linear(rank, text_width, bias=False)),
                                ("ReLU2", nn.ReLU(inplace=True))
                            ])).to(clip_model.dtype)
                    nn.init.constant_(inj_mha_text.Linear1.weight, 1/text_width)
                    nn.init.constant_(inj_mha_text.Linear2.weight, 1/(text_width*rank))

                    if branch == "both":
                        injected_layers = {"visual.mha": {f"{target_layer}": inj_mha_visual},
                                            "text.mha": {f"{target_layer}": inj_mha_text}}

                        if target_layer == "query_key":
                            injected_layers = {"visual.mha": {"query": inj_mha_visual, "key": inj_mha_visual},
                                                "text.mha": {"query": inj_mha_text, "key": inj_mha_text}}
                        if target_layer == "query_value":
                            injected_layers = {"visual.mha": {"query": inj_mha_visual, "value": inj_mha_visual},
                                                "text.mha": {"query": inj_mha_text, "value": inj_mha_text}}
                        if target_layer == "all_in":
                            injected_layers = {"visual.mha": {"query": inj_mha_visual, "key": inj_mha_visual, "value": inj_mha_visual},
                                                "text.mha": {"query": inj_mha_text, "key": inj_mha_text, "value": inj_mha_text}}
                        if target_layer == "all":
                            injected_layers = {"visual.mha": {"query": inj_mha_visual, "key": inj_mha_visual, "value": inj_mha_visual, "out_proj": inj_mha_visual},
                                                "text.mha": {"query": inj_mha_text, "key": inj_mha_text, "value": inj_mha_text, "out_proj": inj_mha_text}}
                    
                    else:
                        target_branch = "visual"
                        inj = inj_mha_visual
                        if branch == "transformer":
                            target_branch = "text"
                            inj = inj_mha_text

                        injected_layers = {f"{target_branch}.mha": {f"{target_layer}": inj}}

                        if target_layer == "query_key":
                            injected_layers = {f"{target_branch}.mha": {"query": inj, "key": inj}}
                        if target_layer == "query_value":
                            injected_layers = {f"{target_branch}.mha": {"query": inj, "value": inj}}
                        if target_layer == "all_in":
                            injected_layers = {f"{target_branch}.mha": {"query": inj, "key": inj, "value": inj}}
                        if target_layer == "all":
                            injected_layers = {f"{target_branch}.mha": {"query": inj, "key": inj, "value": inj, "out_proj": inj}}

                    injected_clip_model = InjectedClip(clip_model, injected_layers).cuda()
                    
                    injected_clip_model.eval()

                    if cfg['dataset'] == "imagenet":
                        print("Preparing ImageNet dataset.")
                        dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

                        val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=64, num_workers=4, shuffle=False)
                        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=4, shuffle=False)

                        train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=4, shuffle=False)
                        train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=4, shuffle=True)
                    else:
                        print("Preparing dataset.")
                        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

                        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
                        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

                        train_tranform = transforms.Compose([
                            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        ])

                        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
                        train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

                    #clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
                    clip_model = finetune_model(cfg, injected_clip_model, train_loader_F, val_loader, dataset, branch)
                    clip_model.eval()

                    # Textual features
                    print("\nGetting textual features as CLIP's classifier.")
                    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

                    # Construct the cache model by few-shot training set
                    print("\nConstructing cache model by few-shot visual features and labels.")
                    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

                    # we do not have to load visual features every time if we only train the textual branch
                    if branch != "transformer" or (seed == seeds[0]):
                        # Pre-load val features
                        print("\nLoading visual features and labels from val set.")
                        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

                        # Pre-load test features
                        print("\nLoading visual features and labels from test set.")
                        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

                    # ------------------------------------------ Tip-Adapter ------------------------------------------
                    lora_acc, tip_adapter_acc = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

                    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
                    tip_adapter_f_acc = run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, 1.0, True)

                    # ------------------------------------------ Tip-Adapter-F Max Loss ------------------------------------------
                    max_loss_acc = run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, 1.0, False)

                    if branch == "both":
                        train_both_branches_lora_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": lora_acc})
                        train_both_branches_tip_adapter_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": tip_adapter_acc})
                        train_both_branches_tip_adapter_f_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": tip_adapter_f_acc})
                        train_both_branches_max_loss_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": max_loss_acc})
                    elif branch == "visual":
                        train_visual_branch_lora_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": lora_acc})
                        train_visual_branch_tip_adapter_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": tip_adapter_acc})
                        train_visual_branch_tip_adapter_f_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": tip_adapter_f_acc})
                        train_visual_branch_max_loss_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": max_loss_acc})
                    else:
                        train_text_branch_lora_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": lora_acc})
                        train_text_branch_tip_adapter_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": tip_adapter_acc})
                        train_text_branch_tip_adapter_f_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": tip_adapter_f_acc})
                        train_text_branch_max_loss_accs.append({"shots": shots, "rank": rank, "target_layer": target_layer, "acc": max_loss_acc})

    res = {"VisualText": train_both_branches_lora_accs,
           "VisualText + Tip-Adapter": train_both_branches_tip_adapter_accs,
           "VisualText + Tip-Adapter-F": train_both_branches_tip_adapter_f_accs,
           "VisualText + MaxLoss": train_both_branches_max_loss_accs,
           "Visual": train_visual_branch_lora_accs,
           "Visual + Tip-Adapter": train_visual_branch_tip_adapter_accs,
           "Visual + Tip-Adapter-F": train_visual_branch_tip_adapter_f_accs,
           "Visual + MaxLoss": train_visual_branch_max_loss_accs,
           "Text": train_text_branch_lora_accs,
           "Text + Tip-Adapter": train_text_branch_tip_adapter_accs,
           "Text + Tip-Adapter-F": train_text_branch_tip_adapter_f_accs,
           "Text + MaxLoss": train_text_branch_max_loss_accs,}

    print(res)
           

if __name__ == '__main__':
    main()
