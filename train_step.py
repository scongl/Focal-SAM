import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from utils import grad_norm



def focal_sam_step(args, model, criterion, inputs, targets, optimizer, cls_num_list, scaler=None):

    if args.prec == "amp":
        with torch.cuda.amp.autocast():
            output = model(inputs)
            loss_ori = criterion(output, targets)
            cls_num_list_tensor = torch.tensor(cls_num_list).cuda()
            sum_cls_num_list = torch.sum(cls_num_list_tensor).cuda()
            coefficients = (1 - cls_num_list_tensor / sum_cls_num_list) ** args.flat_gamma * args.sharpness
            loss = 0.0
            unique_targets = torch.unique(targets)
            idx = torch.arange(targets.size(0)).unsqueeze(1)
            mask = (targets.unsqueeze(1) == unique_targets.unsqueeze(0)).float()
            loss += torch.sum((1 - coefficients[unique_targets]) * loss_ori[idx] * mask)
            loss /= inputs.size(0)

        scaler.scale(loss).backward(retain_graph=True)
        scaler.unscale_(optimizer)
        optimizer.first_step()
        scaler.update()

        with torch.cuda.amp.autocast():
            loss = 0.0
            loss += torch.sum(coefficients[unique_targets] * loss_ori[idx] * mask)
            loss /= inputs.size(0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        optimizer.second_step()
        scaler.update()

        with torch.cuda.amp.autocast():
            output = model(inputs)
            loss = criterion(output, targets)
            loss_sam = 0.0
            loss_sam += torch.sum(coefficients[unique_targets] * loss[idx] * mask)
            loss_sam /= inputs.size(0)

        scaler.scale(loss_sam).backward()
        scaler.unscale_(optimizer)
        optimizer.third_step()
        scaler.update()

        return output, loss.mean()
    
    else:
        output = model(inputs)
        loss_ori = criterion(output, targets)

        # coeffients = [None] * len(cls_num_list)

        # for i in range(len(cls_num_list)):
        #     coeffients[i] = (1 - cls_num_list[i] / sum(cls_num_list)) ** args.flat_gamma * args.sharpness

        cls_num_list_tensor = torch.tensor(cls_num_list).cuda()
        sum_cls_num_list = torch.sum(cls_num_list_tensor).cuda()

        coefficients = (1 - cls_num_list_tensor / sum_cls_num_list) ** args.flat_gamma * args.sharpness

        loss = 0.0
        # for target in torch.unique(targets):
        #     idx = torch.where(targets == target)[0]
        #     loss += (1 - coeffients[target.item()]) * loss_ori[idx].sum()

        unique_targets = torch.unique(targets)
        idx = torch.arange(targets.size(0)).unsqueeze(1)
        mask = (targets.unsqueeze(1) == unique_targets.unsqueeze(0)).float()

        loss += torch.sum((1 - coefficients[unique_targets]) * loss_ori[idx] * mask)

        loss /= inputs.size(0)

        loss.backward(retain_graph=True)
        optimizer.first_step()

        loss = 0.0
        # for target in torch.unique(targets):
        #     idx = torch.where(targets == target)[0]
        #     loss += coeffients[target.item()] * loss_ori[idx].sum()

        loss += torch.sum(coefficients[unique_targets] * loss_ori[idx] * mask)

        loss /= inputs.size(0)

        loss.backward()
        optimizer.second_step()

        output = model(inputs)

        loss = criterion(output, targets)
        loss_sam = 0.0

        # for target in torch.unique(targets):
        #     idx = torch.where(targets == target)[0]
        #     loss_sam += coeffients[target.item()] * loss[idx].sum()

        loss_sam += torch.sum(coefficients[unique_targets] * loss[idx] * mask)

        loss_sam /= inputs.size(0)

        loss_sam.backward()
        optimizer.third_step()

        return output, loss.mean()
