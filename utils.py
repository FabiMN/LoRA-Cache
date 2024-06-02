from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import checkpoint
import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def clip_classifier_with_grad(classnames, template, clip_model):
    clip_weights = []

    #i = 0
    for classname in classnames:
        # Tokenize the prompts
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in template]
        texts = clip.tokenize(texts).cuda()
        # prompt ensemble for ImageNet
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings_normed = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding_mean = class_embeddings_normed.mean(dim=0)
        class_embedding_mean_normed = class_embedding_mean / class_embedding_mean.norm()

        # Checkpoint can be used if there is not enough space on the GPU to store the entire back prop graph
        #class_embedding_mean_normed = checkpoint.checkpoint(text_embeddings_forward, *(clip_model, texts), use_reentrant=False)
        clip_weights.append(class_embedding_mean_normed)

    clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def text_embeddings_forward(clip_model, texts):
    # prompt ensemble for ImageNet
    class_embeddings = clip_model.encode_text(texts)
    class_embeddings_normed = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding_mean = class_embeddings_normed.mean(dim=0)
    class_embedding_mean_normed = class_embedding_mean / class_embedding_mean.norm()
    return class_embedding_mean_normed

def clip_classifier_per_template(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for t in template:
            clip_weights_by_prompt = []
            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                text = [t.format(classname)]
                text = clip.tokenize(text).cuda()
                class_embedding = clip_model.encode_text(text)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                class_embedding = class_embedding.squeeze()
                clip_weights_by_prompt.append(class_embedding)
            clip_weights_by_prompt = torch.stack(clip_weights_by_prompt, dim=1).cuda()
            clip_weights.append(clip_weights_by_prompt)

        clip_weights = torch.stack(clip_weights, dim=2).cuda()
    return clip_weights

def build_cache_model(cfg, clip_model, train_loader_cache, train_loader_cache_extended=None, use_spatial_features=False):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        cache_keys_extended = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    if use_spatial_features:
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        encoding_length = image_features.shape[2]
                        image_features = torch.reshape(image_features.permute(1, 0, 2), (image_features.shape[1], image_features.shape[0]*image_features.shape[2]))
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

                if train_loader_cache_extended:
                    train_features_extended = []
                    for i, (images, target) in enumerate(tqdm(train_loader_cache_extended)):
                        images = images.cuda()
                        image_features = clip_model.encode_image(images)
                        train_features_extended.append(image_features)
                    cache_keys_extended.append(torch.cat(train_features_extended, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)

        # if we use spatial features, cache keys are already normalized
        if not use_spatial_features:
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        #print(f'keys: {cache_keys.shape}')
        if train_loader_cache_extended:
            cache_keys_extended = torch.cat(cache_keys_extended, dim=0).mean(dim=0)
            cache_keys_extended /= cache_keys_extended.norm(dim=-1, keepdim=True)
            #print(f'keys extended: {cache_keys_extended.shape}')

            cache_keys = torch.cat((cache_keys, cache_keys_extended), dim=0)
            cache_values = torch.cat((cache_values, cache_values), dim=0)

        cache_keys = cache_keys.permute(1, 0)

        if train_loader_cache_extended:
            torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots_extended.pt")
            torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots_extended.pt")
        else:
            torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
            torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        if train_loader_cache_extended:
            cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots_extended.pt")
            cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots_extended.pt")
        else:
            cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
            cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader, use_spatial_features=False):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if use_spatial_features:
                    number_spatial_features = image_features.shape[0] - 1
                    encoding_length = image_features.shape[2]
                    image_features = torch.reshape(image_features.permute(1, 0, 2), (image_features.shape[1], image_features.shape[0]*image_features.shape[2]))
                    image_features[:, encoding_length:] = image_features[:, encoding_length:] / number_spatial_features

                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def search_hp_top_k(cfg, cache_keys, cache_values, features, labels, clip_weights, shots, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        k_weight_range = [x * 0.2 for x in range(1, 5)]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        best_alphas = []
        best_count = 0
        best_k = 1
        cutoff = shots//3+2

        if adapter:
            affinity = adapter(features)
        else:
            affinity = features @ cache_keys

        for beta in beta_list:
            for alpha in alpha_list:
                for k in range(1, cutoff):
                    for weight in k_weight_range:
                        aff_vec = affinity.cpu()
                        alphas = []
                        count = 0
                        for row in aff_vec:
                            _indices = np.argsort(row)
                            top_cutoff = _indices[-k:]
                            max_indices = sorted(top_cutoff)
                            if max_indices[k-1] - max_indices[0] >= shots:
                                alphas.append(best_alpha * weight)
                            elif max_indices[0] % shots > max_indices[k-1] % shots:
                                alphas.append(best_alpha * weight)
                            else:
                                alphas.append(best_alpha)
                                count = count + 1
                        
                        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                        clip_logits = 100. * features @ clip_weights
                        
                        tip_logits = clip_logits + torch.mul(cache_logits, torch.FloatTensor(alphas).view(cache_logits.shape[0], 1).cuda())

                        tip_logits = clip_logits + cache_logits * alpha
                        acc = cls_acc(tip_logits, labels)
                    
                        if acc > best_acc:
                            #print("New best setting, beta: {:.2f}, alpha: {:.2f}, k: {}, weight: {:.2f}; accuracy: {:.2f}".format(beta, alpha, k, weight, acc))
                            best_acc = acc
                            best_beta = beta
                            best_alpha = alpha
                            best_alphas = alphas
                            best_count = count
                            best_k = k

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alphas, best_count, best_k

def scale_(x, target):
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y
