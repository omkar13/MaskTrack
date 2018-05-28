# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

__author__ = 'federico perazzi'
__version__ = '2.0.0'

########################################################################
#
# Interface for accessing the DAVIS 2016/2017 dataset.
#
# DAVIS is a video dataset designed for segmentation. The API implemented in
# this file provides functionalities for loading, parsing and visualizing
# images and annotations available in DAVIS. Please visit
# [https://graphics.ethz.ch/~perazzif/davis] for more information on DAVIS,
# including data, paper and supplementary material.
#
########################################################################

import numpy as np

from PIL import Image
from base import Sequence, Annotation, BaseLoader, Segmentation

from config import cfg,phase,db_read_sequences

from easydict import EasyDict as edict

class DAVISLoader(object):
  """
  Helper class for accessing the DAVIS dataset.

  Arguments:
    year          (string): dataset version (2016,2017).
    phase         (string): dataset set eg. train, val. (See config.phase)
    single_object (bool):   assign same id (==1) to each object.

  Members:
    sequences (list): list of 'Sequence' objects containing RGB frames.
    annotations(list): list of 'Annotation' objects containing ground-truth segmentations.
  """
  def __init__(self,year,phase,single_object=False):
    super(DAVISLoader, self).__init__()

    self._year  = year
    self._phase = phase
    self._single_object = single_object

    assert year == "2017" or year == "2016"

    # check the phase
    if year == '2016':
      if not (self._phase == phase.TRAIN or self._phase == phase.VAL or \
          self._phase == phase.TRAINVAL):
            raise Exception("Set \'{}\' not available in DAVIS 2016 ({},{},{})".format(
              self._phase.name,phase.TRAIN.name,phase.VAL.name,phase.TRAINVAL.name))

    # Check single_object if False iif year is 2016
    if self._single_object:
      assert self._year == '2016'

    self._db_sequences = db_read_sequences(year,self._phase)

    # Load sequences
    self.sequences = [Sequence(s.name)
        for s in self._db_sequences]

    # Load sequences
    """
    if self._phase==phase.TESTDEV:
        self.annotations = [None for s in self._db_sequences]
    else:
    """
    self.annotations = [Annotation(s.name,self._single_object) for s in self._db_sequences]

    self._keys = dict(zip([s.name for s in self.sequences],
      range(len(self.sequences))))

    # Check number of frames is correct
    for sequence,db_sequence in zip(self.sequences,self._db_sequences):
      assert len(sequence) == db_sequence.num_frames

    # Check number of annotations is correct
    for annotation,db_sequence in zip(self.sequences,self._db_sequences):
      if (self._phase == phase.TRAIN) or (self._phase == phase.VAL):
        assert len(annotation) == db_sequence.num_frames
      elif self._phase == phase.TESTDEV:
        pass

    try:
      self.color_palette = np.array(Image.open(
        self.annotations[0].files[0]).getpalette()).reshape(-1,3)
    except Exception as e:
      self.color_palette = np.array([[0,255,0]])

  def __len__(self):
    """ Number of sequences."""
    return len(self.sequences)

  def __iter__(self):
    """ Iteratator over pairs of (sequence,annotation)."""
    for sequence,annotation in zip(self.sequences,self.annotations):
      yield sequence,annotation

  def __getitem__(self, key):
    """ Get sequences and annotations pairs."""
    if isinstance(key,str):
      sid = self._keys[key]
    elif isinstance(key,int):
      sid = key
    else:
      raise InputError()

    return edict({
      'images'  : self.sequences[sid],
      'annotations': self.annotations[sid]
      })

  def sequence_name_to_id(self,name):
    """ Map sequence name to index."""
    return self._keys[name]

  def sequence_id_to_name(self,sid):
    """ Map index to sequence name."""
    return self._db_sequences[sid].name

  def iternames(self):
    """ Iterator over sequence names."""
    for s in self._db_sequences:
      yield s.name

  def iteritems(self):
    return self.__iter__()
