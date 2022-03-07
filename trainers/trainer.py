import torch

class Trainer():
    def __init__(self, net, opt, sch):
        self.net=net
        self.opt=opt
        self.sch=sch
        self.epoch=-1

    def load_state_dict( self, state_dict ):
        self.epoch = state_dict['epoch']
        self.net.load_state_dict( state_dict['net'] )
        self.opt.load_state_dict( state_dict['opt'] )
        self.sch.load_state_dict( state_dict['sch'] )

    def state_dict( self ):
        return { 'epoch': self.epoch,
                  'net': self.net.state_dict(),
                  'opt': self.opt.state_dict(),
                  'sch': self.sch.state_dict(),
               }

def cl_train( tra, slog, trainloader, testloader, device, verbose, **cfg):
    net=tra.net
    opt=tra.opt
    sch=tra.sch
    cri=tra.cri

    keys = ['epoch', 'train_cri', 'train_acc', 'test_cri', 'test_acc']
    slog.init_csv( keys )

    epoch0 = tra.epoch+1
    for epoch in range(epoch0,cfg['epochs']):
        tra.epoch = epoch

        train_cri = 0
        train_crr = 0
        train_num = 0

        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            opt.zero_grad()
            outputs = net(inputs)
            loss = cri(outputs, targets)
            loss.backward()
            opt.step()

            train_cri += loss.item()*inputs.shape[0]
            _, predicted = outputs.max(1)
            train_crr += predicted.eq(targets).sum().item()
            train_num += inputs.shape[0]

            if( verbose ):
                print( ' ', epoch, batch_idx, train_cri/train_num, train_crr/train_num )
        sch.step()

        test_cri = 0
        test_crr = 0
        test_num = 0

        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                loss = cri(outputs, targets)

                _, predicted = outputs.max(1)
                test_cri += loss.item()*inputs.shape[0]
                test_crr += predicted.eq(targets).sum().item()
                test_num += inputs.shape[0]

        dlog={}
        dlog['epoch'] = epoch
        dlog['train_cri'] = train_cri/train_num
        dlog['train_acc'] = train_crr/train_num
        dlog['test_cri'] = test_cri/test_num
        dlog['test_acc'] = test_crr/test_num

        slog.write_state_dict( tra.state_dict() )
        slog.append_csv( dlog )
        if( not cfg['noplot'] and epoch > 1 ):
            slog.plot_csv()

    cfg['train_cri'] = dlog['train_cri']
    cfg['train_acc'] = dlog['train_acc']
    cfg['test_cri'] = dlog['test_cri']
    cfg['test_acc'] = dlog['test_acc']

    slog.write_yaml( cfg )

    if( not cfg['nosummary'] ):
        slog.yaml_summary()

    if( not cfg['noplot'] ):
        slog.plot_csvs()

class ClTrainer(Trainer):
    def __init__(self, net, opt, sch, cri):
        super().__init__(net,opt,sch)
        self.cri=cri

    def train(self, logger, trainloader, testloader, device, verbose, **cfg):
        cl_train( self, logger, trainloader, testloader, device, verbose, **cfg)
