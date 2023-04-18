from numpy.core.fromnumeric import transpose
import yaml
import argparse
import time
import copy

import torch
import torchvision
import click

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset
import torchvision.transforms as transforms

from models.extractor import Img2Vec

parser = argparse.ArgumentParser(description='CS7643 final project')
parser.add_argument('--config', default='./configs/config.yaml')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]
    _, pred = torch.max(output, dim=-1)
    correct = pred.eq(target).sum() * 1.0
    acc = correct / batch_size
    return acc


def train(epoch, data_loader, model, optimizer, criterion):

    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    for idx, (data, target) in enumerate(data_loader):
        start = time.time()


        # Probably need to do some kind of tokenization?
        print(target)
        target = [entry[0] for entry in target]
        target = torch.LongTensor(target)

        if torch.cuda.is_available():
            data = data.to("cuda")
            target = target.to("cuda")

        optimizer.zero_grad()  # clear out gradients
        out = model.forward(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()  # gradient descent

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))


def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 10
    cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            out = model.forward(data)
        loss = criterion.forward(out, target)

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_data_cifar():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader

def load_data_coco(batch_size=512):
    """
    CIFAR10 is hosted on torch's repo so when you call torchvision.dataset.CIFAR10, all 
    of the conversions are done for us. For COCO, this dataset is not stored in torch 
    so we have to do everything manually.
    
    """

    # Get paths
    train_data_path = "./data/train2017"
    train_json_path = "./data/annotations/captions_train2017.json"

    validation_data_path = "./data/val2017"
    validation_json_path = "./data/annotations/captions_val2017.json"

    # Load training data
    training_data = torchvision.datasets.CocoCaptions(
        root=train_data_path,
        annFile=train_json_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    # Load validation data
    validation_data = torchvision.datasets.CocoCaptions(
        root=validation_data_path,
        annFile=train_json_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    # Maybe figure out how to transform normalize
    # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py#L89

    # Create DataLoaders
    training_dataloader = torch.utils.data.DataLoader(
        dataset=training_data,
        shuffle=True
    )  
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_data,
        shuffle=True
    )

    return training_dataloader, validation_dataloader

@click.command()
@click.option("--config", required=True)
def main(config):
    global args
    args = parser.parse_args()
    with open(config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    train_loader, test_loader = load_data_coco()

    print(f"train_loader type: {type(train_loader)}")
    print(f"Number of samples: {len(train_loader)}")
    # train_loader type: <class 'torch.utils.data.dataloader.DataLoader'>
    # train_loader type: <class 'torchvision.datasets.coco.CocoCaptions'>

    model = None
    if args.model == 'CNN':
        model = CNN()

    img2vec = Img2Vec(model=args.model)
    model = img2vec.model  # Pretrained model

    # Converting training data to feature vectors using pretrained model. No need to go through training below with CNN part
    for idx, (data, target) in enumerate(train_loader):
        feature_vec = img2vec.get_vec(data)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = None
    if args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.reg)
    best_acc = 0.0
    best_model = None
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)
        # import PIL
        # feature_vector = img2vec.get_vec(PIL.Image.open('./data/rabbit.jpg'))

        # validation loop
        acc, cm = validate(epoch, test_loader, model, criterion)

        # keep the best model
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

    # Save the best weights to a file
    print('Best Prec @1 Acccuracy: {:.4f}'.format(best_acc))
    parameters_str = f"lr={args.learning_rate}-batch_size={args.batch_size}-reg={args.reg}-epochs={args.epochs}-momentum={args.momentum}-loss_type={args.loss}"
    torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '_' + parameters_str + '.pth')


if __name__ == '__main__':
    main()
