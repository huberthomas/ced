# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import argparse
from os.path import isfile, join, isdir
import logging
from helpers import correctFilePath

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

def checkInputParameter(args):
  '''
  @brief Checks the input parameter.
  @param args Input arguments.
  @return Validated arguments in the input structure.
  '''
  if not isfile(args.baseWeights):
    raise IOError('Model "%s" does not exist.' % (args.baseWeights))

  if not isfile(args.netProto):
    raise IOError('Network "%s" does not exist.' % (args.netProto))

  return args

def process(args):
    '''
    @brief Process model solving.
    @param args Input arguments.
    '''
    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()

    solver = caffe.SGDSolver(args.netProto)

    # copy base weights for fine-tuning
    solver.net.copy_from(args.baseWeights)
    solver.step()


def main():
    parser = argparse.ArgumentParser(
        description='Test images on Deep Crisp Boundaries algorithm.')
    parser.add_argument('--gpu', default=False,
                        action='store_true', help='Enable GPU mode.')
    parser.add_argument('--baseWeights', type=str,
                        default='hed_vgg16.caffemodel', help="Path to the Caffe model file.")
    parser.add_argument('--netProto', type=str, default='solver.prototxt',
                        help='Path to the proto network file.')

    try:
      args = checkInputParameter(parser.parse_args())

      process(args)
    except Exception as e:
      logging.error(e)


if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.DEBUG)
  print('#####################')
  print('Deep Crisp Boundaries')
  print('#####################')
  main()
