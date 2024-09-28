import torch
from torch import nn

from models.resnet import ResNet50, WideResNet502

import torchvision

adv_model_clean = ResNet50().cuda()
adv_model_con = WideResNet502().cuda()
adv_model_con.load_state_dict(torch.load("./pgdwrn50_1000.pt"))
adv_model_con = adv_model_con.eval()
adv_model_clean = adv_model_clean.eval()

import numpy as np

#load images to enhance
#x = np.load(, mmap_mode='r')
gen_imgs = x['arr_0']
gt_labels = x['arr_1']



def sample_q(f_cls, gen_images, y, sgld_lr, sgld_std, n_steps):
  """this func takes in replay_buffer now so we have the option to sample from
  scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
  """
  f_cls.eval()
  # get batch size
  bs = 100

  x_k = torch.autograd.Variable(gen_images, requires_grad=True)
  # sgld
  for k in range(n_steps):
      f_prime_cls = torch.autograd.grad(torch.gather(f_cls(x_k), 1, y[:, None]).mean(), [x_k], retain_graph=True)[0]
      x_k.data += sgld_lr * f_prime_cls + sgld_std * torch.randn_like(x_k)
  x_k = torch.clamp(x_k, 0, 1)
  final_samples = x_k.detach()

  return final_samples


from tqdm import tqdm

torch.manual_seed(1234)
boosted_gen_imgs = []
for i in tqdm(range(500)):
  x_batch = gen_imgs[i * 100: (i + 1) * 100]
  x_batch_0_1 = torch.tensor(x_batch / 255.).cuda().permute(0,3,1,2).float()
  with torch.no_grad():
    labels = torch.tensor(gt_labels[i * 100: (i + 1) * 100]).cuda().long()
  b_imgs = sample_q(adv_model_con, x_batch_0_1, labels.long(), 0.2, 1e-4, 15)

  boosted_gen_imgs.append(b_imgs.detach().cpu())
boosted_gen_imgs = torch.cat(boosted_gen_imgs)
boosted_gen_imgs = (boosted_gen_imgs * 255).int()
boosted_gen_imgs = boosted_gen_imgs.detach().cpu().permute(0,2,3,1).numpy().astype(np.uint8)

np.savez("boosted_lang", boosted_gen_imgs)