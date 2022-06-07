import torch

ckpt = torch.load('runs/CCP/best.pt')

ckpt['encoder_params']['pretrained_model_cfg'] = 'data/ccp'

ckpt['encoder_params'].pop('normalization')
ckpt['encoder_params'].pop('similarity')
ckpt['encoder_params'].pop('pretrained_file')
ckpt['encoder_params'].pop('projection_dim')

torch.save(ckpt, 'runs/CCP/best.pt')