from torch.optim import *
from torch.optim.lr_scheduler import *

#################################################
def _gen_opt( _params_, name, keys, **cfg ):
    args = {}
    for k, v in cfg.items():
        if( k[:4] == 'opt_' ):
            k = k[4:]
        if( k in keys ):
            args[k]=v
    return eval( '{}( _params_, **args )'.format(name) )


def gen_opt( _params_, name=None, **cfg ):
    if( name is None ):
        name = cfg['opt']

    name = name.lower()
    if( name == 'sgd' ):
        keys = ['lr', 'momentum', 'weight_decay', 'nesterov']
        opt = _gen_opt( _params_, 'SGD', keys, **cfg )

    elif( name == 'adam' ):
        keys = ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
        opt = _gen_opt( _params_, 'Adam', keys, **cfg )

    else:
        raise NotImplementedError(name)

    return opt

#################################################
def _gen_sch( _opt_, name, keys, **cfg ):
    args = {}
    for k, v in cfg.items():
        if( k[:4] == 'sch_' ):
            k = k[4:]
        if( k in keys ):
            args[k]=v
    return eval( '{}( _opt_, **args )'.format(name) )

def gen_sch( _opt_, **cfg ):
    name = None
    if( name is None ):
        name = cfg['sch']

    name = name.lower()
    if( name == 'coslr' or name == 'cos' or name == "cosineannealinglr" ):
        keys = ['T_max', 'eta_min', 'last_epoch', 'verbose']
        sch = _gen_sch( _opt_, "CosineAnnealingLR", keys, **cfg )

    elif( name == 'steplr' or name == 'step' ):
        keys = ['step_size', 'gamma', 'last_epoch', 'verbose']
        sch = _gen_sch( _opt_, "StepLR", keys, **cfg )

    else:
        raise NotImplementedError(name)

    return sch

if( __name__ == '__main__' ):
    import torch.nn as nn

    cfg = {}
    cfg['model'] = 'VGG'
    cfg['opt'] = 'sgd'
    cfg['lr'] = 0.1
    cfg['opt_momentum'] = 0.9

    conv = nn.Linear( 4, 4 )
    opt = gen_opt( conv.parameters(), **cfg )
    print( opt )

    cfg['betas'] = [0.9, 0.99]
    opt = gen_opt( conv.parameters(), 'Adam', **cfg )
    print( opt )

    cfg['sch'] = 'Cos'
    cfg['T_max'] = 100
    sch = gen_sch( opt, **cfg )
    print( sch )

