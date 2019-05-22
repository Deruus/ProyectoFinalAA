#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from time import strftime
from flask import Flask, request, redirect, url_for
from flask import render_template, send_from_directory
from flask_bootstrap import Bootstrap

import PIL.Image
import numpy as np
import tensorflow as tf

from dream import *
import random
import math

UPLOAD_FOLDER = 'carga/'
RESULT_FOLDER = 'resultado/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
bootstrap = Bootstrap(app)


def allowed_file(filename):
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return False


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            extension = file.filename.rsplit('.', 1)[1].lower()
            filename = strftime('%Y%m%d%H%M%S') + '.' + extension
            fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fullpath)
            filename = 'd' + filename
            dream(fullpath, filename)
            return redirect(url_for('download', filename=filename))
    return render_template('index.html')


def dream(fullpath, filename):
    img0 = PIL.Image.open(fullpath).convert('RGB')
    img0 = np.float32(img0)



    layer_tensor = model.layer_tensors[7][:, :, :, 0:3]
    img0 = optimizacion_recursiva(layer_tensor=layer_tensor, image=img0,
                                      num_iterations=10, step_size=3.0, rescale_factor=0.7,
                                      num_repeats=4, blend=0.2)
    
    #img0 = optimizar_imagen(layer_tensor, img0, num_iterations=10, step_size=6.0, tile_size=400)
    
    
    #img0 = optimizacion_recursiva(layer_tensor=layer_tensor, image=img0,
    #                                  num_iterations=10, step_size=3.0, rescale_factor=0.7,
    #                                  num_repeats=4, blend=0.2)
    #layer_tensor = model.layer_tensors[3]
    #img0 = optimizacion_recursiva(layer_tensor=layer_tensor, image=img0,
    #                                  num_iterations=10, step_size=3.0, rescale_factor=0.7,
    #                                  num_repeats=4, blend=0.2)
    #layer_tensor = model.layer_tensors[4]
    #img0 = optimizacion_recursiva(layer_tensor=layer_tensor, image=img0,
    #                                  num_iterations=10, step_size=3.0, rescale_factor=0.7,
    #                                  num_repeats=4, blend=0.2)
    #layer_tensor = model.layer_tensors[7][:, :, :, 0:3]
    #optimizacion_recursiva(layer_tensor=layer_tensor, image=img0,
    #                           num_iterations=10, step_size=3.0, rescale_factor=0.7,
    #                           num_repeats=4, blend=0.2)



    guardar_imagen(img0, filename=os.path.join(app.config['RESULT_FOLDER'], filename))



@app.route('/dream/<filename>')
def download(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run()
