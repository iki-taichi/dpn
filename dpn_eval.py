# coding: utf-8
#
# Taichi Iki (2016-06-15)
# dpn_eval.py
# deep prednetのtheanoによる実装(評価の方)

# 主要な関数
#   - load
#   - applyimg

# 使用法
#   モデルを読み込んで入力画像をもとに内部状態を変換

import itertools
from PIL import Image
from PIL import ImageOps
import os
import pickle
import numpy as np
import theano
from theano import tensor

floatX = np.float32

class model:
    def showh(self):
        u'''validation errorの履歴を表示'''
        for x in self.arg['history']: print x[0], x[1]
        
    def showch(self):
        u'''train errorの履歴を表示'''
        for x in self.arg['costhistory']: print x[1], x[2]
    
    u'''dpn_eval.load(modelpath)の戻り値'''
    def resetstate(self):
        u'''内部状態E, R, recA, C, Aを0に初期化'''
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
        u'''計算に使用'''
        x = np.maximum(x, 0)
        x = x * (x > 0)
        return (abs(x) + x) / 2
        
    def sigmoid(self, x):
        u'''計算に使用'''
        return 1.0 / (1.0 + np.exp(-x))

    def Upsample(self, x):
        u'''計算に使用; axis2, 3を2倍に拡大(nearest-neighbor)'''
        t = np.repeat(x, 2, axis=2)
        return np.repeat(t, 2, axis=3)
        
    def spad2d(self, x, plast=1, psecondlast=1):
        u'''計算に使用; 最後と最後から2番目の次元の両端に指定個0を挿入'''
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
        u'''計算に使用; max poolingで大きさを1/2にする'''
        xx = tensor.TensorType(theano.config.floatX, (False,)*len(x.shape))()
        f = theano.function([xx], tensor.signal.pool.pool_2d(xx, (2, 2), ignore_border=True))
        return f(x)
    
    def ConvLSTM(self, param, layerid, layerinfo, E, R, C, RPP, padw):
        u'''計算に使用; ConvLSTMの計算'''
        si = str(layerid)
        ch, h, w = layerinfo
        zi = tensor.nnet.conv.conv2d(self.spad2d(E, padw, padw), param['cl_WE_'+si][0]).eval()
        zi += tensor.nnet.conv.conv2d(self.spad2d(R, padw, padw), param['cl_WR_'+si][0]).eval()
        zi += C*(param['cl_WC_'+si][0])
        zi += param['cl_b_'+si][0]
    
        zf = tensor.nnet.conv.conv2d(self.spad2d(E, padw, padw), param['cl_WE_'+si][1]).eval()
        zf += tensor.nnet.conv.conv2d(self.spad2d(R, padw, padw), param['cl_WR_'+si][1]).eval()
        zf += C*(param['cl_WC_'+si][1])
        zf += param['cl_b_'+si][1]
    
        zc = tensor.nnet.conv.conv2d(self.spad2d(E, padw, padw), param['cl_WE_'+si][2]).eval()
        zc += tensor.nnet.conv.conv2d(self.spad2d(R, padw, padw), param['cl_WR_'+si][2]).eval()
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
    
    def pathtoitensor(self, imgpath):
        u'''画像のパスをうけてnumpy.arrayを返す'''
        imgch, imgh, imgw = self.imgshape
        img = Image.open(imgpath)
        img = img.resize((imgw, imgh))
        if imgch == 3:
            itensor = np.array(img.convert('RGB'))
        else:
            itensor = np.array(ImageOps.grayscale(img))
        return itensor
    
    def itensortotesnor(self, itensor):
        u'''itensor[h, w(, ch)] 0-255 -> tensor[ch, h, w] 0.0-1.0'''
        t = np.array(itensor, dtype=floatX)
        t /= 255.0
        if len(t.shape) == 2:
            t = t.reshape((t.shape[0], t.shape[1], 1))
        t = t.swapaxes(0, 2).swapaxes(1, 2)
        t = t.reshape((1, t.shape[0], t.shape[1], t.shape[2])) 
        return  t
    
    def tensortoitensor(self, t):
        u'''tensor[ch, h, w] 0.0-1.0 -> itensor[h, w, ch] 0-255'''
        it = np.array(t)
        it = it.reshape((t.shape[1], t.shape[2], t.shape[3]))
        it = it.swapaxes(1, 2).swapaxes(0, 2)
        if it.shape[2] == 1:
            it = it.reshape((it.shape[0], it.shape[1]))
        it *= 255
        it = np.array(it, dtype='uint8')
        return  it
    def ttoit(self, t):
        return self.tensortoitensor(t)
    
    def procConvLSTM(self, R=None, C=None):
        u'''内部状態E, R, Cを使って計算し, 引数のR, Cに新しい状態を入れる'''
        li = self.arg['layerinfo']
        ll = len(li)
        padw = (self.arg['kernelsize'] - 1) // 2
        
        if R is None: R = self.R
        if C is None: C = self.C
        
        for i in reversed(range(0, ll)):
            R[i], C[i] = self.ConvLSTM(self.param, i, li[i], self.E[i], self.R[i], self.C[i], 
                None if i == ll-1 else self.Upsample(self.R[i+1]), padw)
    
    def procRecA(self, recA=None):
        u'''内部状態Rを使って計算し, 引数のrecAに新しい状態を入れる'''
        li = self.arg['layerinfo']
        ll = len(li)
        padw = (self.arg['kernelsize'] - 1) // 2
        
        if recA is None: recA = self.recA
        
        for i in range(0, ll):
            tmp = tensor.nnet.conv.conv2d(self.spad2d(self.R[i], padw, padw), self.param['cr_W_'+str(i)]).eval()
            tmp += self.param['cr_b_'+str(i)][None, :, None, None]
            recA[i] = self.relu(tmp)
        recA[0] = np.minimum(recA[0], 1.0)
    
    def procEandA(self, x, E=None, A=None):
        u'''入力xと内部状態recAを使って計算し, 引数のE, Aに新しい状態を入れる'''
        li = self.arg['layerinfo']
        ll = len(li)
        padw = (self.arg['kernelsize'] - 1) // 2
        
        if E is None: E = self.E
        if A is None: A = self.A
        
        A[0] = x
        for i in range(0, ll):
            ch, h, w = li[i]
            e1 = self.relu(self.recA[i] - A[i])
            e2 = self.relu(A[i] - self.recA[i])
            E[i] = np.concatenate([e1, e2], axis=1)
            if i != ll-1:
                tmp = tensor.nnet.conv.conv2d(self.spad2d(E[i], padw, padw), self.param['cu_W_'+str(i)]).eval()
                tmp += self.param['cu_b_'+str(i)][None, :, None, None]
                A[i+1] = self.MaxPool(tmp)
    
    def applyimg(self, x):
        u'''画像をモデルに作用させる
        x: 画像パスまたは画像をnumpy.arrayにしたもの
        出力: 事前に予想された画像
        * 内部状態が変わる
        '''
        
        # 入力を変換する
        if type(x) == str or type(x) == unicode:
            x = self.pathtoitensor(x)
        x = self.itensortotesnor(x)
        
        self.procConvLSTM()
        self.procRecA()
        self.procEandA(x)
        
        return self.tensortoitensor(self.recA[0])
    
    def predextra(self, seedimgpathlist, outputcount=0, show=True, savedir=None):
        u'''種となる画像のseqを受け取って画像列を保存'''
        if outputcount <= 0: outputcount = self.arg['timesteplen']
        if savedir is not None:
            if not os.path.exists(savedir): os.mkdir(savedir)
            if not savedir.endswith('/'): savedir += '/'
        self.resetstate()
        factor = 1.0*self.imgshape[0]*self.imgshape[1]*self.imgshape[2]
        print 'sum E[0] > 0.0:', (self.E[0]>0.0).sum()/factor, ', sum E[0]:', self.E[0].sum()/factor
        count = 0
        for p in seedimgpathlist:
            arr = self.applyimg(p)
            img = Image.fromarray(arr)
            if savedir: img.save(savedir + str(count) + '.png')
            if show: img.show()
            count += 1
            print '%d / %d done'%(count, outputcount), 'sum E[0] > 0.0:', (self.E[0]>0.0).sum()/factor, ', sum E[0]:', self.E[0].sum()/factor
            if count >= outputcount: break
        
        for i in range(0, outputcount - count):
            arr = self.applyimg(arr)
            img = Image.fromarray(arr)
            if savedir: img.save(savedir + str(count) + '.png')
            if show: img.show()
            count += 1
            print '%d / %d done'%(count, outputcount), 'sum E[0] > 0.0:', (self.E[0]>0.0).sum()/factor, ', sum E[0]:', self.E[0].sum()/factor

def load(modelname):
    a = model()
    a.zippath = u'./' + modelname + u'.npz'
    a.pklpath = u'./' + modelname + u'.pkl'
    a.pklbpath = u'./' + modelname + u'.pklb'
    if os.path.exists(a.pklbpath):
        with open(a.pklbpath, 'rb') as f: a.arg = pickle.Unpickler(f).load()
    else:
        with open(a.pklpath, 'r') as f: a.arg = pickle.Unpickler(f).load()
    npz = np.load(a.zippath)
    a.param = {}
    for k, v in npz.items(): a.param[k] = v
    a.imgshape = (a.arg['imgch'], a.arg['imgh'], a.arg['imgw'])
    a.resetstate()
    
    return a
