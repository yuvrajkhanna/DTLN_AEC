from DTLN_model import DTLN_model
import argparse
from pkg_resources import parse_version
import tensorflow as tf
quantization = True
weights_file='weights/DTLN_model.h5'
target_folder='weights/model'
converter = DTLN_model()
converter.create_tf_lite_model(weights_file, 
                                  target_folder,
                                  norm_stft=True,
                                  use_dynamic_range_quant=bool(quantization))
