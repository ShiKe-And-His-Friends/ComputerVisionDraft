import os
import torch
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train ,model ,yolo_loss ,loss_history ,eval_callback ,optimizer ,epoch ,epoch_step,epoch_step_val,gen ,gen_val \
                  ,Epoch ,cuda ,fp16 ,scaler ,save_period ,save_dir ,local_rank = 0):

    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step ,desc=f'Epoch{epoch + 1}/{Epoch}' ,postfix=dict ,mininterval=0.3)
    model_train.train()

    for iteration,batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images ,targets = batch[0] ,batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        # --------------------------------#
        #  清零梯度
        # --------------------------------#
        optimizer.zero_grad()
        if not fp16:
            # --------------------------------#
            #  前向传播
            # --------------------------------#
            outputs = model_train(images)
            loss_value_all = 0
            # --------------------------------#
            #  计算损失
            # --------------------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l,outputs[l] ,targets)
                loss_value_all += loss_item
            loss_value = loss_value_all
            # --------------------------------#
            #  反向传播
            # --------------------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # --------------------------------#
                #  前向传播
                # --------------------------------#
                outputs = model_train(images)
                loss_value_all = 0
                # --------------------------------#
                #  计算损失
                # --------------------------------#
                for l in range(len(outputs)):
                    with torch.cuda.amp.autocast(enabled=False):
                        predication = outputs[l].float()
                    loss_item = yolo_loss(l ,predication ,targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all
            # --------------------------------#
            #  反向传播
            # --------------------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()
        if (local_rank == 0) :
            pbar.set_postfix(**{'loss' : loss / (iteration + 1), 'lr' : get_lr(optimizer)})
            pbar.update(1)

    print('Train Done')
    return None