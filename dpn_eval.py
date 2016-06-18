# coding: utf-8
#
# Taichi Iki (2016-06-15)
# dpn_eval.py
# deep prednetのtheanoによる実装(評価の方)

# 主要な関数
#   - load モデルファイルのロード
#   - applyimg 画像をモデルに適用
#   - predextra 画像seqを種に再構築画像/予想画像を生成

# 使用法
#   モデルを読み込んで入力画像をもとに内部状態を変換

import itertools
from PIL import Image 
import os
import pickle
import numpy as np
import theano
from theano import tensor

floatX = np.float32

class model:
    def resetstate(self):
        li = self.arg['layerinfo']
        self.E = []
        self.R = []
        self.recA = []
        self.C = []
        self.A = []
        for i in range(0, len(li)):
            ch, h, w = li[i]
            # Errorは+/-で2倍
            self.E.append(np.zeros((1, 2*ch, h, w), floatX))
            self.R.append(np.zeros((1, ch, h, w), floatX))
            self.recA.append(np.zeros((1, ch, h, w), floatX))
            self.C.append(np.zeros((1, ch, h, w), floatX))
            self.A.append(np.zeros((1, ch, h, w), floatX))
    
    def relu(self, x):
        x = np.maximum(x, 0)
        x = x * (x > 0)
        return (abs(x) + x) / 2
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def Upsample(self, x):
        t = np.repeat(x, 2, axis=2)
        return np.repeat(t, 2, axis=3)
        
    def spad2d(self, x, plast=1, psecondlast=1):
        input_shape = x.shape
        output_shape = list(input_shape)
        output_shape[-1] += 2* plast
        output_shape[-2] += 2* psecondlast
        output = np.zeros(tuple(output_shape), floatX)
        indices = [slice(None)]*(len(output_shape) -2)
        indices += [slice(psecondlast, input_shape[-2] + psecondlast)]
        indices += [slice(plast, input_shape[-1] + plast)]
        xx = tensor.TensorType(theano.config.floatX, (False,)*len(x.shape))()
        xy = tensor.TensorType(theano.config.floatX, (False,)*len(x.shape))()
        f = theano.function([xx, xy], tensor.set_subtensor(xx[tuple(indices)], xy))
        return f(output, x)
        
    def MaxPool(self, x):
        xx = tensor.TensorType(theano.config.floatX, (False,)*len(x.shape))()
        f = theano.function([xx], tensor.signal.pool.pool_2d(xx, (2, 2), ignore_border=True))
        return f(x)
    
    def ConvLSTM(self, param, layerid, layerinfo, E, R, C, RPP, padw):
        si = str(layerid)
        ch, h, w = layerinfo
        zi = tensor.nnet.conv.conv2d(self.spad2d(E, padw, padw), param['cl_WE_'+si][0]).eval()
        zi += tensor.nnet.conv.conv2d(self.spad2d(R, padw, padw), param['cl_WR_'+si][0]).eval()
        zi += C*(param['cl_WC_'+si][0])
        zi += param['cl_b_'+si][0]
    
        zf = tensor.nnet.conv.conv2d(self.spad2d(E, padw, padw), param['cl_WE_'+si][1]).eval()
        zf += tensor.nnet.conv.conv2d(self.spad2d(R, padw, padw), param['cl_WR_'+si][1]).eval()
        zf += param['cl_b_'+si][1]
    
        zc = tensor.nnet.conv.conv2d(self.spad2d(E, padw, padw), param['cl_WE_'+si][2]).eval()
        zc += tensor.nnet.conv.conv2d(self.spad2d(R, padw, padw), param['cl_WR_'+si][2]).eval()
        zc += C*(param['cl_WC_'+si][2])
        zc += param['cl_b_'+si][2]
    
        zo = tensor.nnet.conv.conv2d(self.spad2d(E, padw, padw), param['cl_WE_'+si][3]).eval()
        zo += tensor.nnet.conv.conv2d(self.spad2d(R, padw, padw), param['cl_WR_'+si][3]).eval()
        zo += C*(param['cl_WC_'+si][3])
        zo += param['cl_b_'+si][3]
    
        if RPP is not None:
            zi += tensor.nnet.conv.conv2d(self.spad2d(RPP, padw, padw), param['cl_WRPP_'+si][0]).eval()
            zf += tensor.nnet.conv.conv2d(self.spad2d(RPP, padw, padw), param['cl_WRPP_'+si][1]).eval()
            zc += tensor.nnet.conv.conv2d(self.spad2d(RPP, padw, padw), param['cl_WRPP_'+si][2]).eval()
            zo += tensor.nnet.conv.conv2d(self.spad2d(RPP, padw, padw), param['cl_WRPP_'+si][3]).eval()
    
        i = self.sigmoid(zi)
        f = self.sigmoid(zf)
        Cnext = f*C + np.tanh(zc)
        o = self.sigmoid(zo)
        Rnext = o * np.tanh(Cnext)
        return (Rnext, Cnext)
    
    # x: [ch, h, w], 0.0 - 1.0 に正規化して入れる
    def applyimg(self, x):
        if type(x) == str or type(x) == unicode:
            x = self.ptom(x)
        x = np.array(x.reshape((1, x.shape[0],x.shape[1],x.shape[2])), dtype=floatX)

        li = self.arg['layerinfo']
        ll = len(li)
        padw = (self.arg['kernelsize'] - 1) // 2
        
        E = self.E
        R = self.R
        recA = self.recA
        C = self.C
        A = self.A
        for i in reversed(range(0, ll)):
            R[i], C[i] = self.ConvLSTM(self.param, i, li[i], E[i], R[i], C[i], None if i == ll-1 else self.Upsample(R[i+1]), padw)
        
        for i in range(0, ll):
            tmp = tensor.nnet.conv.conv2d(self.spad2d(R[i], padw, padw), self.param['cr_W_'+str(i)]).eval()
            tmp += self.param['cr_b_'+str(i)][None, :, None, None]
            recA[i] = self.relu(tmp)
        recA[0] = np.minimum(recA[0], 1.0)
        
        A = x
        for i in range(0, ll):
            ch, h, w = li[i]
            e1 = self.relu(recA[i] - A)
            e2 = self.relu(A - recA[i])
            E[i] = np.concatenate([e1, e2], axis=1)
            if i != ll-1:
                tmp = tensor.nnet.conv.conv2d(self.spad2d(E[i], padw, padw), self.param['cu_W_'+str(i)]).eval()
                tmp += self.param['cu_b_'+str(i)][None, :, None, None]
                A = self.MaxPool(tmp)
        return recA[0]

    def showh(self):
        for x in self.arg['history']: print x[0], x[1]
        
    def showch(self):
        for x in self.arg['costhistory']: print x[1], x[2]

    def imgshape(self):
        return (self.arg['imgch'], self.arg['imgh'], self.arg['imgw'])

    def ptom(self, imgpath):
        imgch, imgh, imgw = self.imgshape
        img = Image.open(imgpath)
        img = img.resize((imgw, imgh)) 
        if imgch == 3:
            img = np.array(img.convert('RGB'))
            img = img.swapaxes(0, 2).swapaxes(1, 2)
            # shape [h, w, ch] -> [ch, h, W]
        else:
            img = np.array(ImageOps.grayscale(img))
            img = img.reshape((1, imgh, imgw))
        img = img / 255.0
        return img

    def forcv(self, x=None):
        if x is None:
            x = self.recA[0]
        s = x.shape
        return np.array(x.reshape((s[1], s[2], s[3])).swapaxes(0,2).swapaxes(0,1)*255, dtype='uint8')
        
    def toapply(self, x):
        s = x.shape
        return np.array(x.swapaxes(0,1).swapaxes(0,2), dtype=floatX)/255.0
        
    def predextra(self, seedimgpathlist, outputcount=0, show=True, savedir=None):
        if outputcount <= 0: outputcount = self.arg['timesteplen']
        if savedir is not None:
            if not os.path.exists(savedir): os.mkdir(savedir)
            if not savedir.endswith('/'): savedir += '/'
        self.resetstate()
        
        count = 0
        for p in seedimgpathlist:
            arr = self.forcv(self.applyimg(p))
            img = Image.fromarray(arr)
            if savedir: img.save(savedir + str(count) + '.png')
            if show: img.show()
            count += 1
            print '%d / %d done'%(count, outputcount)
        
        for i in range(0, outputcount - count):
            arr = self.forcv(self.applyimg(self.toapply(arr)))
            img = Image.fromarray(arr)
            if savedir: img.save(savedir + str(count) + '.png')
            if show: img.show()
            count += 1
            print '%d / %d done'%(count, outputcount)

def load(modelname):
    a = model()
    a.zippath = u'./' + modelname + u'.npz'
    a.pklpath = u'./' + modelname + u'.pkl'
    with open(a.pklpath, 'r') as f: a.arg = pickle.Unpickler(f).load()
    npz = np.load(a.zippath)
    a.param = {}
    for k, v in npz.items(): a.param[k] = v
    a.imgshape = (a.arg['imgch'], a.arg['imgh'], a.arg['imgw'])
    a.resetstate()
    
    return a
