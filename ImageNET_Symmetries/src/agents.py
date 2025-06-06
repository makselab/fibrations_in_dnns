import torch

from torch.nn.functional import cross_entropy
from torch.optim import SGD
from convGnT import ConvGnT
from resGnT import ResGnT
from GnT import GnT
from breaksymm import BreakSymmetry

LIST_OPTS = {"SGD":SGD}

class agent_model(object):
    def __init__(self, net, optimizer, agent):

        self.net = net
        self.opt = LIST_OPTS[optimizer['name']](self.net.parameters(), **optimizer['cfg'])
        self.name = agent['name']
        self.loss_func = cross_entropy
        self.previous_features = None

        LIST_LEARN = {'BP': self.learn_bp,
                      'MLCBP': self.learn_cbp,
                      'CBP': self.learn_cbp,
                      'Res-CBP': self.learn_cbp,
                      'BreakSymm': self.learn_symmetry}

        LIST_GNT = {'BP': None,
                    'MLCBP': GnT,
                    'CBP': ConvGnT,
                    'Res-CBP': ResGnT,
                    'BreakSymm': BreakSymmetry}

        if self.name == 'BP':
            self.gnt = None

        if self.name == 'CBP':
            self.gnt = LIST_GNT[self.name](
                                            net=self.net,
                                            hidden_activation=self.net.act_type,
                                            opt=self.opt,
                                            num_last_filter_outputs=net.last_filter_output,
                                            **agent['cfg'])
        if self.name in ['MLCBP', 'Res-CBP', 'BreakSymm']:
            self.gnt = LIST_GNT[self.name](
                                            net=self.net,
                                            hidden_activation=self.net.act_type,
                                            opt=self.opt,
                                            **agent['cfg'])

        self.learn = LIST_LEARN[self.name]

    def learn_bp(self, x, target, classes=None):
        self.opt.zero_grad()
        output, features = self.net(x=x)
        if classes is not None: output = output[:,classes]

        loss = self.loss_func(output, target)
        #self.previous_features = features

        loss.backward()
        self.opt.step()

        return loss.detach(), output.detach()

    def learn_cbp(self, x, target, classes=None):
        self.opt.zero_grad()
        output, features = self.net(x=x)
        if classes is not None: output = output[:,classes]

        loss = self.loss_func(output, target)
        self.previous_features = features

        loss.backward()
        self.opt.step()
        self.gnt.gen_and_test(features=self.previous_features)

        return loss.detach(), output.detach()

    def learn_symmetry(self,x,target, classes=None):
        self.opt.zero_grad()
        output, features = self.net(x=x)
        if classes is not None: output = output[:,classes]

        loss = self.loss_func(output, target)

        loss.backward()
        self.opt.step()

        self.gnt.gen_and_test()

        return loss.detach(), output.detach()

    #   current_features = [] if self.use_cbp else None
    #   if self.use_cbp: self.resgnt.gen_and_test(current_features)