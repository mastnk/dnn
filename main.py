import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

from utils import *
from datasets.cifar import cifar10_loaders
from models.utils import gen_model
from trainers.trainer import *
from trainers.utils import *

########################################################################
parser = argparse.ArgumentParser(description='')

parser.add_argument('--clear', action='store_true', default=False)
parser.add_argument('--nolog', action='store_true', default=False)
parser.add_argument('--config', default=None)
parser.add_argument('--verbose', action='store_true', default=False)

parser.add_argument('--log_dir', default='checkpoint' )
parser.add_argument('--log_name', default='0' )

parser.add_argument('--data_root', default='/datasets/' )
parser.add_argument('--data_num', type=int, default=0 )

parser.add_argument('--noplot', action='store_true', default=False)
parser.add_argument('--nosummary', action='store_true', default=False)


parser.add_argument('--opt', default='Adam' )
parser.add_argument('--opt_lr', type=float, default=0.001 )

parser.add_argument('--sch', default='CosLR' )
parser.add_argument('--sch_gamma', type=float, default=0.5 )
parser.add_argument('--sch_step_size', type=float, default=50 )

parser.add_argument('--model', default='VGG11' )
parser.add_argument('--batch_size', type=int, default=128 )
parser.add_argument('--epochs', type=int, default=200 )


########################################################################
cfg = vars( parser.parse_args() )

# pop from cfg, they are not saved in yaml
clear = cfg.pop('clear')
nolog = cfg.pop('nolog')
config = cfg.pop('config')
verbose = cfg.pop('verbose')

log_dir = cfg.pop('log_dir')
log_name = cfg.pop('log_name')

data_root = cfg.pop('data_root')

slog = SimpleLogger( log_dir, nolog, log_name = log_name )
print( 'log dir: ', slog.dir_name )

cfg_file = {}
if( config is not None ):
    if( os.path.exists( config ) ):
        with open( config, 'r') as yin:
            cfg_file = yaml.load(yin, Loader=yaml.SafeLoader)
        print( 'read {}'.format(config) )

for k in sorted(cfg.keys()):
    if( k in cfg_file.keys() ):
        cfg[k] = cfg_file[k]
        print( 'file:', k, cfg[k] )
    else:
        print( 'args:', k, cfg[k] )

if( clear ):
    slog.clear_log()
    print( 'clear log' )

slog.write_yaml( cfg )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

net = gen_model( cfg['model'], **cfg )
net = net.to(device)

epoch = -1
opt = gen_opt( net.parameters(), **cfg )

cfg['sch_T_max'] = cfg['epochs']
cfg['sch_last_epoch'] = epoch
sch = gen_sch( opt, **cfg )

cri = nn.CrossEntropyLoss()


trainer = ClTrainer( net, opt, sch, cri )
state = slog.read_state_dict()
if( state is  None ):
    slog.write_state_dict( trainer.state_dict() )
else:
    trainer.load_state_dict( state )
    print( 'read state' )

trainloader, testloader = cifar10_loaders(batch_size=cfg['batch_size'], data_root=data_root, data_num=cfg['data_num'])

trainer.train( slog, trainloader, testloader, device, verbose, **cfg )

