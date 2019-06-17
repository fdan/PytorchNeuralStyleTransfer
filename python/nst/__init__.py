#! /usr/bin/env python

import subprocess
from optparse import OptionParser
from timeit import default_timer as timer
import cProfile

import memory_profiler

import torch
from torch.autograd import Variable # deprecated - use Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

from PIL import Image

from matplotlib import pyplot


MODEL = '../../Models/vgg_conv.pth'

TOTAL_GPU_MEMORY = 0


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


def postp(tensor):  # to clip results in the range [0,1]
    """
    :param tensor: torch.Tensor
    :return: PIL.Image
    """

    # what's this do?
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])

    # what's this do?
    postpb = transforms.Compose([transforms.ToPILImage()])

    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


def image_to_tensor(image, do_cuda):
    """
    :param [PIL.Image]
    :return: [torch.Tensor]
    """
    # pre and post processing for images
    img_size = 512

    tforms = transforms.Compose([transforms.Resize(img_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])

    tensor = tforms(image)

    if do_cuda:
        return tensor.unsqueeze(0).cuda()
    else:
        return tensor.unsqueeze(0)


def doit():

    p = OptionParser()
    p.add_option("", "--content", dest='content', action='store')
    p.add_option("", "--style", dest='style', action='store')
    p.add_option("", "--output", dest='output', action='store')
    p.add_option("", "--iterations", dest='iterations', action='store')
    p.add_option("", "--loss", dest='loss', action='store')
    p.add_option("", "--engine", dest="engine", action='store', default='gpu')

    opts, args = p.parse_args()

    style = opts.style
    content = opts.content
    output = opts.output
    engine = opts.engine
    iterations = opts.iterations
    max_loss = opts.loss

    if iterations and max_loss:
        print "iterations and max_loss canot both be specified"
        return

    if engine == 'gpu':
        if torch.cuda.is_available():
            do_cuda = True
            smi_cmd = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader']
            total_gpu_memory = subprocess.check_output(smi_cmd).rstrip('\n')
            print "using cuda\navailable memory: %s" % total_gpu_memory
        else:
            msg = "gpu mode was requested, but cuda is not available"
            raise RuntimeError(msg)

    elif engine == 'cpu':
        do_cuda = False
        from psutil import virtual_memory
        mem = virtual_memory()
        # mem.total
        print "using cpu\navailable memory: %s Gb" % (mem.total / 1000000000)

    else:
        msg = "invalid arg for engine: valid options are cpu, gpu"
        raise RuntimeError(msg)


    # get network
    vgg = VGG()

    vgg.load_state_dict(torch.load(MODEL))

    for param in vgg.parameters():
        param.requires_grad = False

    if do_cuda:
        vgg.cuda()

    # list of PIL images
    input_images = [Image.open(style), Image.open(content)]

    style_image = Image.open(style)
    content_image = Image.open(content)

    style_tensor = image_to_tensor(style_image, do_cuda)
    content_tensor = image_to_tensor(content_image, do_cuda)

    # variable is dperecated
    opt_img = Variable(content_tensor.data.clone(), requires_grad=True)

    # define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

    if do_cuda:
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    # these are good weights settings:
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    # compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_tensor, style_layers)]
    content_targets = [A.detach() for A in vgg(content_tensor, content_layers)]
    targets = style_targets + content_targets

    # run style transfer

    show_iter = 20
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]
    current_loss = [9999999]

    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        current_loss[0] = loss.item()
        n_iter[0] += 1
        if n_iter[0] % show_iter == (show_iter - 1):
            nice_loss = '{:,.0f}'.format(loss.item())
            max_mem_cached = torch.cuda.max_memory_cached(0) / 1000000
            msg = ''
            msg += 'Iteration: %d, ' % (n_iter[0] + 1)
            msg += 'loss: %s, ' % (nice_loss)

            if do_cuda:
                msg += 'memory used: %s of %s' % (max_mem_cached, total_gpu_memory)
            else:
                mem_usage = memory_profiler.memory_usage(proc=-1, interval=0.1, timeout=0.1)
                msg += 'memory used: %.02f of %s Gb' % (mem_usage[0]/1000, mem.total/1000000000)

            print msg
        return loss

    if iterations:
        max_iter = int(iterations)
        print ''
        while n_iter[0] <= max_iter:
            optimizer.step(closure)
        print ''

    if max_loss:
        while current_loss[0] > int(max_loss):
            optimizer.step(closure)

    out_img = postp(opt_img.data[0].cpu().squeeze())

    if output:
        out_img.save(output)
    else:
        pyplot.imshow(out_img)


def main():

    # global TOTAL_GPU_MEMORY
    # smi_cmd = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader']
    # TOTAL_GPU_MEMORY = subprocess.check_output(smi_cmd).rstrip('\n')

    print ''
    start = timer()

    # print 1, memory_profiler.memory_usage(proc=-1, interval=1, timeout=1)

    doit()

    end = timer()
    duration = "%.02f seconds" % float(end-start)

    print 'completed\n'
    print "duration:", duration

    max_mem_cached = torch.cuda.max_memory_cached(0)/1000000
    # print "memory used: %s of %s\n" % (max_mem_cached, TOTAL_GPU_MEMORY)


if __name__ == "__main__":
    main()