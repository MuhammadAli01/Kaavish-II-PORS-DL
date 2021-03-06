# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config import *
from utils import *
from data import Fashion_attr_prediction, Fashion_inshop
from net import f_model
from in_shop_eval import eval


data_transform_train = transforms.Compose([
    transforms.Scale(IMG_SIZE),
    transforms.RandomSizedCrop(CROP_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transform_test = transforms.Compose([
    transforms.Scale(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


train_loader = torch.utils.data.DataLoader(
    Fashion_attr_prediction(type="train", transform=data_transform_train),
    batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
print(f"len(train_loader): {len(train_loader)}")

test_loader = torch.utils.data.DataLoader(
    Fashion_attr_prediction(type="test", transform=data_transform_test),
    batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

triplet_loader = torch.utils.data.DataLoader(
    Fashion_attr_prediction(type="triplet", transform=data_transform_train),
    batch_size=TRIPLET_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

if ENABLE_INSHOP_DATASET:
    triplet_in_shop_loader = torch.utils.data.DataLoader(
        Fashion_inshop(type="train", transform=data_transform_train),
        batch_size=TRIPLET_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

model = f_model(freeze_param=FREEZE_PARAM, model_path=DUMPED_MODEL).cuda(GPU_ID)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=MOMENTUM)

if DUMPED_MODEL:
    start_epoch = int(DUMPED_MODEL.split('/')[-1].split('_')[0]) + 1
else:
    start_epoch = 1
# start_epoch = 1
# print(f"start_epoch: {start_epoch}")

# if not DUMPED_MODEL:
#     writer = SummaryWriter(log_dir=f"runs/f'freeze={FREEZE_PARAM}'/50_categories/inshop={ENABLE_INSHOP_DATASET}/lr={LR}/{EPOCH}epochs/{datetime.now().strftime('%b%d_%H-%M-%S')}")
# else:
#     writer = SummaryWriter(
#         log_dir=f"runs/f'freeze={FREEZE_PARAM}'/{CATEGORIES}_categories/inshop={ENABLE_INSHOP_DATASET}/lr={LR}/{EPOCH}epochs/{DUMPED_MODEL}")

# writer = SummaryWriter(log_dir=f"runs/freeze={FREEZE_PARAM}/{CATEGORIES}_categories/inshop={ENABLE_INSHOP_DATASET}"
#                                f"/lr={LR}")
writer = SummaryWriter(log_dir=f"runs/freeze={FREEZE_PARAM}/{CATEGORIES}_categories/inshop={ENABLE_INSHOP_DATASET}"
                               f"/lr={LR}/with_eastern")


def train(epoch):
    model.train()
    criterion_c = nn.CrossEntropyLoss()
    if ENABLE_TRIPLET_WITH_COSINE:
        criterion_t = TripletMarginLossCosine()
    else:
        criterion_t = nn.TripletMarginLoss()
    triplet_loader_iter = iter(triplet_loader)
    triplet_type = 0
    if ENABLE_INSHOP_DATASET:
        triplet_in_shop_loader_iter = iter(triplet_in_shop_loader)

    TEST_INTERVAL = len(train_loader)
    DUMP_INTERVAL = TEST_INTERVAL

    running_loss = 0.0
    if TRIPLET_WEIGHT:
        running_clf_loss = 0.0
        running_triplet_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(GPU_ID), target.cuda(GPU_ID)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        outputs = model(data)[0]
        classification_loss = criterion_c(outputs, target)

        pred = outputs.data.max(1, keepdim=True)[1]  # Tensor of dim (test_batch_size, 1)
        running_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if TRIPLET_WEIGHT:
            if ENABLE_INSHOP_DATASET and random.random() < INSHOP_DATASET_PRECENT:
                triplet_type = 1
                try:
                    data_tri_list = next(triplet_in_shop_loader_iter)
                except StopIteration:
                    triplet_in_shop_loader_iter = iter(triplet_in_shop_loader)
                    data_tri_list = next(triplet_in_shop_loader_iter)
            else:
                triplet_type = 0
                try:
                    data_tri_list = next(triplet_loader_iter)
                except StopIteration:
                    triplet_loader_iter = iter(triplet_loader)
                    data_tri_list = next(triplet_loader_iter)
            triplet_batch_size = data_tri_list[0].shape[0]
            data_tri = torch.cat(data_tri_list, 0)
            data_tri = data_tri.cuda(GPU_ID)
            data_tri = Variable(data_tri, requires_grad=True)
            feats = model(data_tri)[1]
            triplet_loss = criterion_t(
                feats[:triplet_batch_size],
                feats[triplet_batch_size:2 * triplet_batch_size],
                feats[2 * triplet_batch_size:]
            )
            loss = classification_loss + triplet_loss * TRIPLET_WEIGHT
        else:
            loss = classification_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.data * TRAIN_BATCH_SIZE
        if TRIPLET_WEIGHT:
            running_clf_loss += classification_loss.data.item() * TRAIN_BATCH_SIZE
            running_triplet_loss += triplet_loss.data.item() * TRIPLET_BATCH_SIZE

        step_no = (epoch - 1) * len(train_loader) + batch_idx

        if batch_idx % LOG_INTERVAL == 0:
            if TRIPLET_WEIGHT:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAll Loss: {:.4f}\t'
                      'Triple Loss({}): {:.4f}\tClassification Loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    # 100. * batch_idx / len(train_loader), loss.data[0], triplet_type,
                    100. * batch_idx / len(train_loader), loss.data, triplet_type,
                    # triplet_loss.data[0], classification_loss.data[0]))
                    triplet_loss.data.item(), classification_loss.data.item()))

                if ENABLE_INSHOP_DATASET:
                    writer.add_scalar('Loss/train/triplet',
                                      running_triplet_loss / (LOG_INTERVAL * TRIPLET_BATCH_SIZE * INSHOP_DATASET_PRECENT),
                                      step_no)
                else:
                    writer.add_scalar('Loss/train/triplet',
                                      running_triplet_loss / (LOG_INTERVAL * TRIPLET_BATCH_SIZE), step_no)

                writer.add_scalar('Loss/train/classification',
                                  running_clf_loss / (LOG_INTERVAL * TRAIN_BATCH_SIZE),
                                  step_no)
                running_triplet_loss, running_clf_loss = 0.0, 0.0

            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tClassification Loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))

            writer.add_scalar('Loss/train',
                              running_loss / (LOG_INTERVAL * TRAIN_BATCH_SIZE),
                              step_no)
            writer.add_scalar('Accuracy/train',
                              float(100. * running_correct / (LOG_INTERVAL * TRAIN_BATCH_SIZE)),
                              step_no)
            running_loss, running_correct = 0.0, 0

        if batch_idx % TEST_INTERVAL == 0:
            # print(f'Test() called at step_no: {step_no}')
            test(step_no, full=True)

        if batch_idx and batch_idx % DUMP_INTERVAL == 0:
            print('Model saved to {}'.format(dump_model(model, epoch, batch_idx)))

    print('Model saved to {}'.format(dump_model(model, epoch)))


def test(step_no, full=False):
    model.eval()  # Tells model you are testing
    # print(f"len(test_loader): {len(test_loader)}")
    # criterion = nn.CrossEntropyLoss(size_average=False)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0

    if full:
        test_batch_count = len(test_loader)
    else:
        test_batch_count = TEST_BATCH_COUNT

    for batch_idx, (data, target) in enumerate(test_loader):
        # print(f'batch_idx: {batch_idx}')
        data, target = data.cuda(GPU_ID), target.cuda(GPU_ID)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)[0]  # Tensor of dim (test_batch_size, num_classes))
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]  # Tensor of dim (test_batch_size, 1)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx > test_batch_count:
            break
    test_loss /= (test_batch_count * TEST_BATCH_SIZE)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(test_loss), correct, (test_batch_count * TEST_BATCH_SIZE),
        float(100. * correct / (test_batch_count * TEST_BATCH_SIZE))))

    writer.add_scalar('Loss/test', test_loss, step_no)
    writer.add_scalar('Accuracy/test',
                      float(100. * correct / (test_batch_count * TEST_BATCH_SIZE)),
                      step_no)


def get_conf_matrix():
    model.eval()  # Tells model that you're testing
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    print(f'dataset: {test_loader.dataset}')
    print(f'dataset len: {len(test_loader.dataset)}')

    all_preds = torch.tensor([])
    all_targets = torch.tensor([])
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 5 == 0:
            print(f'batch_idx: {batch_idx}') 
        data, target = data.cuda(GPU_ID), target.cuda(GPU_ID)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)[0]  # Tensor of dim (test_batch_size, num_classes)
        # print(f'data.item: {criterion(output, target).data.item()}, {type(criterion(output, target).data.item())}')
        # test_loss += criterion(output, target).data[0]
        # print(f'output shape: {output.shape}')
        # print(f'target shape: {target.shape}')
        # print(f'output: {output}')
        # print(f'target: {target}')  # 1D tensor of dim (num_classes)
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]  # Tensor of dim (test_batch_size, 1)

        all_preds = torch.cat(
            (all_preds, pred.to(device='cpu', dtype=torch.float32))
            ,dim=0
        )
        all_targets = torch.cat(
            (all_targets, target.to(device='cpu', dtype=torch.float32))
            ,dim=0
        )

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # if batch_idx > TEST_BATCH_COUNT:
        #     break
    # print(f'all_preds: {all_preds}')
    # print(f'all_targets: {all_targets}')
    # print(f'all_preds shape: {all_preds.shape}')
    # print(f'all_targets shape: {all_targets.shape}')

    cm = confusion_matrix(all_targets, all_preds.squeeze())

    test_loss /= ((batch_idx + 1) * TEST_BATCH_SIZE)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(test_loss), correct, ((batch_idx + 1) * TEST_BATCH_SIZE),
        float(100. * correct / ((batch_idx + 1) * TEST_BATCH_SIZE))))
    return cm


if __name__ == "__main__":
    for epoch in range(start_epoch, EPOCH + 1):
        train(epoch)
