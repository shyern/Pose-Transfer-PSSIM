#!/usr/bin/python3
'''
Dump a caffemodel without the help of original prototxt

@ref https://github.com/apache/incubator-mxnet/blob/master/tools/caffe_converter/caffe_parser.py
@ref https://github.com/apache/incubator-mxnet/blob/master/tools/caffe_converter/convert_model.py
'''

import caffe.proto.caffe_pb2 as caffe_pb2
import sys
import os

if len(sys.argv) < 2: raise Exception("missing arg")
fpath = sys.argv[1]
if not os.path.exists(fpath): raise Exception("file missing")
caffemodel = open(fpath, 'rb').read()

model = caffe_pb2.NetParameter()
model.ParseFromString(caffemodel)

if len(model.layer)>0:
    layers = model.layer
elif len(model.layers)>0:
    layers = model.layers
else:
    raise Exception("Unexpected input")

for l in layers:
    print("\x1b[1;31mname\x1b[0;m: {}".format(l.name))
    print("    \x1b[1;33mtype\x1b[;m: {}".format(l.type))
    print("    \x1b[1;33mbottom\x1b[;m: ", end='')
    for blob in l.bottom:
        print(blob, end=' ')
    print('')
    print("    \x1b[1;33mtop\x1b[;m: ", end='')
    for blob in l.top:
        print(blob, end=' ')
    print('')

print('note, please explore the layer attributes in ipython interactive mode.')
