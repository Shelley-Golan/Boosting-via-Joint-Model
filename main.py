import argparse


import wandb

from torchvision.utils import make_grid
from tqdm import tqdm
from data import get_data
from models.resnet import ResNet50, WideResNet502
import torch
from adversaries import L2PGDAttack
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from accelerate import Accelerator

def binary_loss_fake(logits, targets):
    selected_logits = logits[range(logits.size(0)), targets]
    return torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, torch.zeros_like(selected_logits))


def binary_loss_clean(logits, targets):
    selected_logits = logits[range(logits.size(0)), targets]
    return torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, torch.ones_like(selected_logits))


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(args):
    # Seed
    seed_everything(42)

    # Get data loaders
    dload_train, dload_valid, dload_test, dload_fake = get_data(imagenet_path=args.imagenet_path, batch_size=128)



    # Get model
    if args.arch == 'rn50':
        model = ResNet50()
        model.model.fc = torch.nn.Linear(2048, 1000)
        model = model.cuda()
    if args.arch == 'widern502':
        model = WideResNet502()
        model.model.fc = torch.nn.Linear(2048, 1000)
        model = model.cuda()

    # Get optimizer
    params = model.parameters()
    if args.optimizer == "adam":
        optim = torch.optim.Adam(params, lr=args.lr, betas=[0.9, .999], weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optim = torch.optim.AdamW(params, lr=args.lr, betas=[.9, .95], weight_decay=args.weight_decay)
    else:
        optim = torch.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)
    # Get Attacker
    attacker = L2PGDAttack(predict=model, eps=100., nb_iter=5, rand_init=False, eps_iter=0.1, loss_fn=binary_loss_fake, early_stop=True)
    attacker_real = L2PGDAttack(predict=model, eps=1., nb_iter=2, rand_init=True, eps_iter=0.5,
                                   loss_fn=binary_loss_clean)  # random init
    eval_attacker = attacker

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[4000], gamma=0.1)
    accelerator = Accelerator(log_with="wandb")
    # Start training
    scaler = GradScaler()
    cur_iter = 0
    it = iter(dload_fake)

    global_fake, global_labels_fake = next(it)




    if args.wandb == 0:
        wandb.login()
        # Do wandb logging if applicable
        accelerator.init_trackers(
            # set the wandb project where this run will be logged
            project_name="",
            config=args,
            init_kwargs={"wandb": {"entity": ""}}
        )


    model, optim, dload_train, dload_fake, dload_valid, attacker, attacker_real = accelerator.prepare(
         model, optim, dload_train, dload_fake, dload_valid, attacker, attacker_real
    )


    device = accelerator.device
    torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(args.n_epochs)):
        pbar = tqdm(total=len(dload_train), desc=f"Training epoch {epoch}")

        for batch_id, (images, labels) in enumerate(dload_train):
            if cur_iter % 1000 == 999:
                attacker.nb_iter += 1

            images, labels = images.to(model.device), labels.to(model.device)
            # Get fake images
            images_fake, labels_fake = next(it)
            images_fake = images_fake.to(model.device)
            labels_fake = labels_fake.to(model.device).long()

            images_real = images  # images[(args.batch_size // 2):]
            labels_real = labels  # labels[(args.batch_size // 2):]

        with autocast():
            model.eval()
            x_adv_clean, delta_adv = attacker_real.perturb(images_real, labels_real)
            model.train()
            #
            model.eval()
            x_adv_fake, delta_adv = attacker.perturb(images_fake, labels_fake, x_adv_clean)
            model.train()


            with autocast():
                logits_adv_fake = model(x_adv_fake)
                loss_adv_fake = binary_loss_fake(logits_adv_fake.clone().detach(), labels_fake)
                logits_fake = model(images_fake)
                loss_fake = binary_loss_fake(logits_fake, labels_fake)
                logits_real = model(images_real)
                loss_real = binary_loss_clean(logits_real, labels_real)
                logits_real_adv = model(x_adv_clean)
                loss_real_adv = binary_loss_clean(logits_real_adv.clone().detach(), labels_real)
                loss_cls = nn.CrossEntropyLoss()(logits_real_adv, labels_real) + nn.CrossEntropyLoss()(logits_adv_fake.clone().detach(), labels_fake)
                loss = loss_real_adv + loss_adv_fake + loss_cls # loss_real + loss_aug_adv

                acc_real = (logits_real_adv.max(1)[1] == labels_real).float().mean()
                acc_fake = (logits_adv_fake.max(1)[1] == labels_fake).float().mean()

            if args.wandb == 0:
                accelerator.log({"loss": loss.item()}, step=cur_iter)
                accelerator.log({"acc_real": acc_real}, step=cur_iter)
                accelerator.log({"acc_fake": acc_fake}, step=cur_iter)
                accelerator.log({"loss_adv_fake": loss_adv_fake.item()}, step=cur_iter)
                accelerator.log({"loss_fake": loss_fake.item()}, step=cur_iter)
                accelerator.log({"loss_real": loss_real.item()}, step=cur_iter)
                accelerator.log({"loss_real_adv": loss_real_adv.item()}, step=cur_iter)
                accelerator.log({"loss_cls": loss_cls.item()}, step=cur_iter)

                proba_real = torch.sigmoid(logits_real[range(logits_real.size(0)), labels_real]).mean()
                proba_real_adv = torch.sigmoid(logits_real_adv[range(logits_real_adv.size(0)), labels_real]).mean()
                proba_fake_adv = torch.sigmoid(logits_adv_fake[range(logits_adv_fake.size(0)), labels_fake]).mean()
                proba_fake = torch.sigmoid(logits_fake[range(logits_fake.size(0)), labels_fake]).mean()
                accelerator.log({"proba_real": proba_real}, step=cur_iter)
                accelerator.log({"proba_real_adv": proba_real_adv}, step=cur_iter)
                accelerator.log({"proba_fake_adv": proba_fake_adv}, step=cur_iter)
                accelerator.log({"proba_fake": proba_fake}, step=cur_iter)
            # Scale the loss for the backward pass
            optim.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            # Unscales the gradients and calls optimizer.step()
            optim.step()

            # Updates the scale for next iteration
            scheduler.step(epoch=cur_iter)
            pbar.set_postfix(loss=loss.item())
            pbar.update()
            if cur_iter % 100 == 0:
                model.eval()
                probs = 0

                # Eval
                for batch_id, (eval_images, eval_labels) in enumerate(dload_valid):
                    eval_images, eval_labels = eval_images.to(model.device), eval_labels.to(model.device)
                    x_adv_valid, delta_adv = attacker_real.perturb(eval_images, eval_labels)
                    logits_adv_valid = model(x_adv_valid)
                    loss_adv_valid = binary_loss_clean(logits_adv_valid, eval_labels)
                    loss_cls_valid = nn.CrossEntropyLoss()(logits_adv_valid, eval_labels)
                    acc_valid = (logits_adv_valid.max(1)[1] == eval_labels).float().mean()
                    proba_valid = torch.sigmoid(logits_adv_valid[range(logits_adv_valid.size(0)), eval_labels]).mean()
                accelerator.log({"proba_valid": proba_valid}, step=cur_iter)
                accelerator.log({"loss_bce_valid": loss_adv_valid.item()}, step=cur_iter)
                accelerator.log({"loss_cls_valid": loss_cls_valid.item()}, step=cur_iter)
                accelerator.log({"acc_valid": acc_valid}, step=cur_iter)
                x_adv_fake_global, delta_adv = eval_attacker.perturb(global_fake.clone().to(model.device), global_labels_fake.to(model.device).long())
                adv_0_1 = torch.clamp(x_adv_fake_global, 0, 1).detach().cpu()
                grid_of_images = make_grid(adv_0_1, nrow=4, normalize=False)


                if args.wandb == 0:
                    accelerator.log({"eval": wandb.Image(grid_of_images)}, step=cur_iter)

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), f"./pgdwrn50_{cur_iter}.pt")
            cur_iter += 1
        pbar.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # optimization
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd", "adamw"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    #
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--imagenet_path", type=str, default="")
    parser.add_argument("--arch", type=str, default='widern502', choices=["dino, rn50, convnext, widern502, widern1012, efficientnet"])
    args = parser.parse_args()
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
