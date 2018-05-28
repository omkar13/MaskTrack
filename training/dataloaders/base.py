# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

import functools
import os.path as osp

import numpy as np

from PIL import Image
from skimage.io import ImageCollection

from config import cfg
from io import imread_indexed,imwrite_indexed

#################################
# HELPER FUNCTIONS
#################################

def _load_annotation(filename,single_object):
  """ Load image given filename."""

  annotation,_ = imread_indexed(filename)

  if single_object:
    annotation = (annotation != 0).astype(np.uint8)

  return annotation

def _get_num_objects(annotation):
  """ Count number of objects from segmentation mask"""

  ids = sorted(np.unique(annotation))

  # Remove unknown-label
  ids = ids[:-1] if ids[-1] == 255 else ids

  # Handle no-background case
  ids = ids if ids[0] else ids[1:]

  return len(ids)

#################################
# LOADER CLASSES
#################################

class BaseLoader(ImageCollection):

  """
  Base class to load image sets (inherit from skimage.ImageCollection).

  Arguments:
    path      (string): path to sequence folder.
    regex     (string): regular expression to define image search pattern.
    load_func (func)  : function to load image from disk (see skimage.ImageCollection).

  """

  def __init__(self,path,regex,load_func=None):
    super(BaseLoader, self).__init__(
        osp.join(path + '/' + regex),load_func=load_func)

    # Sequence name
    self.name = osp.basename(path)

    # Check sequence name
    if not self.name in cfg.SEQUENCES:
        raise Exception("Sequence name \'{}\' not found.".format(self.name))

    # Check sequence length
    if len(self) != cfg.SEQUENCES[self.name].num_frames:
      raise Exception("Incorrect frames number for sequence" +
          " \'{}\': found {}, expected {}.".format(
            self.name,len(self),cfg.SEQUENCES[self.name].num_frames))

  def __str__(self):
    return "< class: '{}' name: '{}', frames: {} >".format(
        type(self).__name__,self.name,len(self))

class Sequence(BaseLoader):

  """
  Load image sequences.

  Arguments:
    name  (string): sequence name.
    regex (string): regular expression to define image search pattern.

  """

  def __init__(self,name,regex="*.jpg"):
    super(Sequence, self).__init__(
        osp.join(cfg.PATH.SEQUENCES,name),regex)

class Segmentation(BaseLoader):

  """
  Load image sequences.

  Arguments:
    path          (string): path to sequence folder.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self,path,single_object,regex="*.png"):
    super(Segmentation, self).__init__(path,regex,
       functools.partial(_load_annotation,single_object=single_object))

    if len(self):
      # Extract color palette from image file
      self.color_palette = Image.open(self.files[0]).getpalette()

      if self.color_palette is not None:
        self.color_palette = np.array(
            self.color_palette).reshape(-1,3)
      else:
        self.color_palette = np.array([[255.0,0,0]])
    else:
      self.color_palette = np.array([])

    self.n_objects = _get_num_objects(self[0])

  def iter_objects_id(self):
    """
    Iterate over objects providing object id for each of them.
    """
    for obj_id in range(1,self.n_objects+1):
      yield obj_id

  def iter_objects(self):
    """
    Iterate over objects providing binary masks for each of them.
    """

    for obj_id in self.iter_objects_id():
      bn_segmentation = [(s==obj_id).astype(np.uint8) for s in self]
      yield bn_segmentation

class Annotation(Segmentation):

  """
  Load ground-truth annotations.

  Arguments:
    name          (string): sequence name.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self,name,single_object,regex="*.png"):
    super(Annotation, self).__init__(
       osp.join(cfg.PATH.ANNOTATIONS,name),single_object,regex)


class Deformation(ImageCollection):

  def __init__(self, name, regex):
    #print(osp.join(cfg.PATH.DEFORMATIONS, name,regex))
    super(Deformation, self).__init__(osp.join(cfg.PATH.DEFORMATIONS, name,regex))