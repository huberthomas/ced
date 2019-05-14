# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
import scipy.io
import os
import sys
import argparse
import cv2
import logging
import time
from os.path import join, splitext, split, isfile
from helpers import correctFilePath, getImageFileNames

# add Caffe to Python path
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

def checkInputParameter(args):
  '''
  @brief Checks the input parameter.
  @param args Input arguments.
  @return Validated arguments in the input structure.
  '''
  if args.inputDir is None or len(args.inputDir) == 0:
    raise IOError('Input directory is undefined.')

  args.inputDir = correctFilePath(args.inputDir)

  if args.outputDir is None:
    parent = os.path.split(os.path.abspath(args.inputDir))[0]
    args.outputDir = os.path.join(parent, os.path.basename(
        os.path.dirname(args.inputDir)) + '_ced')

    if args.ms:
      args.outputDir = args.outputDir + '_multiscale'

    if not os.path.exists(args.outputDir):
      os.makedirs(args.outputDir)

  else:
    if args.outputDir is None or len(args.outputDir) == 0:
      raise IOError('Output directory is undefined.')

  args.outputDir = correctFilePath(args.outputDir)

  if not isfile(args.model):
    raise IOError('Model "%s" does not exist.' % (args.model))

  if not isfile(args.netProto):
    raise IOError('Network "%s" does not exist.' % (args.netProto))

  return args


def plotSingleScale(scaleList, size):
  '''
  @brief Plot single scale image in a figure.
  @param size Size of the figure.
  '''
  pylab.rcParams['figure.figsize'] = size, size / 2

  plt.figure()
  for i in range(0, len(scaleList)):
      s = plt.subplot(1, 5, i+1)
      plt.imshow(1-scaleList[i], cmap=cm.Greys_r)
      s.set_xticklabels([])
      s.set_yticklabels([])
      s.yaxis.set_ticks_position('none')
      s.xaxis.set_ticks_position('none')
  plt.tight_layout()

def initCaffe(protoNetPath, caffeModelPath, useGpu = False):
  '''
  @brief Initialize Caffe.
  @param protoNetPath Path to the prototxt file.
  @param caffeModelPath Path to the Caffe model.
  @return Loaded Caffe network.
  '''
  if useGpu:
    caffe.set_mode_gpu()
    caffe.set_device(0)
  else:
    caffe.set_mode_cpu()

  return caffe.Net(protoNetPath, caffeModelPath, caffe.TEST) 

def forward(net, data):
  '''
  @brief Forward data through the network.
  @param net Caffe initialized network.
  @param data Image data.
  @return Network forward result output.
  '''
  assert data.ndim == 3
  data -= np.array((104.00698793, 116.66876762, 122.67891434))
  data = data.transpose((2, 0, 1))
  # shape for input (data blob is N x C x H x W), set data
  net.blobs['data'].reshape(1, *data.shape)
  net.blobs['data'].data[...] = data
  # run net and take argmax for prediction
  return net.forward()

def processInputDir(args):
  '''
  @brief 
  '''
  print(args.outputDir)

  inputDir = args.inputDir
  imgFileNames = getImageFileNames(inputDir)

  net = initCaffe(args.netProto, args.model, args.gpu)

  numImg = len(imgFileNames)

  for i in range(numImg):
    start = time.time()

    filename = imgFileNames[i]

    img = cv2.imread(join(inputDir, filename)).astype(np.float32)
    
    if img.ndim == 2:
      img = img[:, :, np.newaxis]
      img = np.repeat(img, 3, 2)

    h, w, c = img.shape

    scales = [1]
    
    if args.ms:
      scales += [0.5, 1.5]
      scales.sort()

    edge = np.zeros((h, w), np.float32)

    for s in scales:

      if s != 1:
        sH = int(s * h)
        sW = int(s * w)
      
        sImg = cv2.resize(img, (sW, sH), interpolation=cv2.INTER_CUBIC).astype(np.float32)
      else:
        sImg = img

      # run net and get prediction
      res = forward(net, sImg)

      sEdge = np.squeeze(res[args.outputLayer][0, 0, :, :])
      edge += cv2.resize(sEdge, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


    if len(scales) > 1:
      edge /= len(scales)

    # normalize values
    if edge.max() > 0:
      edge /= edge.max()

    end = time.time()
    diffTime = end-start
    logging.info('%d/%d: %s [%f s]' % (i + 1, numImg, filename, diffTime))

    #cv2.imshow('%s' % (filename), edge)
    #cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #scale_lst = [edge]
    #plotSingleScale(scale_lst, 22)

    fn, ext = splitext(filename)
    scipy.misc.imsave(join(args.outputDir, fn + '.png'), edge)

    if args.saveMat:
      scipy.io.savemat(join(args.outputDir, fn), dict({'edge': edge / edge.max()}), appendmat=True)
    #cv2.imwrite(join(args.outputDir, fn + '_cv.png'), edge)
    exit(0)
    

def main():
    '''
    @brief Modified from https://github.com/s9xie/hed/blob/master/examples/hed/HED-tutorial.ipynb
    '''
    parser = argparse.ArgumentParser(
        description='Test images on Deep Crisp Boundaries algorithm.')
    parser.add_argument('-i', '--inputDir', type=str, required=True,
                        help='Input image directory.')
    parser.add_argument('-o' ,'--outputDir', type=str, default=None,
                        help='Processed image output directory.')
    parser.add_argument('--gpu', default=False, action='store_true', help='Enable GPU mode.')
    parser.add_argument('--saveMat', default=False, action='store_true', help='Save output to Matlab file.')
    parser.add_argument('--ms', default=False, action='store_true', help='Use multiscale.')
    parser.add_argument('--model', type=str,
                        default='hed_vgg16.caffemodel', help="Path to the Caffe model file.")
    parser.add_argument('--netProto', type=str, default='deploy.prototxt',
                        help='Path to the proto network file.')
    parser.add_argument('--outputLayer', type=str,
                        default='out_put', help='Output layer.')

    try:      
      args = checkInputParameter(parser.parse_args())

      processInputDir(args)      
    except Exception as e:
      logging.error(e)

if __name__ == '__main__':
  logging.basicConfig(
      format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.DEBUG)
  print('#####################')
  print('Deep Crisp Boundaries')
  print('#####################')
  main()
