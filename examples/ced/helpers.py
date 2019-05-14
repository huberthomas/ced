# -*- coding: utf-8 -*-
import os
from os.path import splitext

def correctFilePath(path):
  '''
  @brief correct file separators and add separator for directories at the end.
  '''
  path = path.replace('/', os.path.sep)
  path = path.replace('\\', os.path.sep)

  if os.path.isdir(path):
    if path[len(path) - 1] != os.path.sep:
      path = path + os.path.sep

  return path

def getImageFileNames(inputDir, supportedExtensions=['.png', '.jpg', '.jpeg']):
  '''
  @brief Get image files (png, jpg, jpeg) from an input directory.
  @param inputDir Input directory that contains images.
  @param supportedExtensions Only files with supported extensions are included in the final list.
  @return List of images file names.
  '''
  res = []

  for root, directories, files in os.walk(inputDir):
      for f in files:
          for extension in supportedExtensions:
            fn, ext = splitext(f.lower())

            if extension == ext:
              res.append(f)

  return res
