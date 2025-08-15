import argparse
from misc.cfg import CFG as cfg
import torch
import torch.optim as optim
import os
from tqdm import tqdm
from condense_hyot import load_resized_data
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from model_dist import define_model


def train_model(model, train_loader, args):
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[2 * args.pretrain_epochs // 3, 5 * args.pretrain_epochs // 6],
        gamma=0.2,
    )

    for epoch in range(0, args.pretrain_epochs):
        print(f"\nEpoch {epoch + 1}/{args.pretrain_epochs}")
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Iter", leave=False)):
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # model save
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(model.state_dict(), args.save_dir + '{}.pkl'.format(args.identity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--demo", action='store_true', help='for debugging, do not save results')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()
    args.identity = args.dataset + "-" + args.net_type + "-ipc" + str(args.ipc)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_dir = "checkpoints/"
    args.pretrain_epochs = 100

    trainset, val_loader = load_resized_data(args)
    train_loader = MultiEpochsDataLoader(trainset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)
    nclass = trainset.nclass
    model = define_model(args, nclass).to(args.device)
    train_model(model, train_loader, args)
