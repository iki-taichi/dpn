# coding: utf-8
#
# Taichi Iki (2016-06-15)
# dpn_learn.py
# deep prednetのtheanoによる実装
# 下で公開されているLSTMのコードをベースに作成
# http://deeplearning.net/tutorial/lstm.html

# 主要な関数
#   - get_sampleset: サンプルをロード
#   - construct_model: モデルを定義 
#   - train_network: 学習ループ

# 使用法
#   まずサンプルをメモリーにロード,　整理
#   モデルを構築(メタパラメータは"modelname".pklとしてpickleされる)
#   定期的に"modelname".npzとして学習したパラメータを保存
#   dpn_eval.pyで"modelname"をロードして予想を行う
#   パラメタの設定はこのスクリプトの一番下のargで行う

#   - optimizerがadamではなくadadelta

import pickle
import numpy as np
import sys
import time
import os

from PIL import Image
from PIL import ImageOps

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config
import re

config.exception_verbosity = 'high'


def floatX_array(data):
    return np.asarray(data, dtype=config.floatX)

def get_sampleset(arg):
    imgw = arg['imgw']
    imgh = arg['imgh']
    imgch = arg['imgch']
    tslen = arg['timesteplen']
    
    sampleset = []
    for dirpath in arg['samplesetpath']:
        if not dirpath.endswith('/'): dirpath += '/'
        print 'sampleset from ', dirpath
        
        imgcount = 0
        seq = []
        files = [(x, re.search('(\d+)',x)) for x in os.listdir(dirpath)]
        files = map(lambda x: (x[0], int(x[1].group(0))), filter(lambda x:x[1], files))
        for t in sorted(files, lambda x, y:x[1]-y[1]):
            fn = t[0]
            img = Image.open(dirpath + fn)
            img = img.resize((imgw, imgh)) 
            if imgch == 3:
                img = floatX_array(img.convert('RGB'))
                img = img.swapaxes(0, 2).swapaxes(1, 2)
                # shape [h, w, ch] -> [ch, h, W]
            else:
                img = floatX_array(ImageOps.grayscale(img))
                img = img.reshape((1, imgh, imgw))
            img = img / 255.0
            seq.append(img)
            imgcount += 1
            if imgcount % tslen == 0:
                sampleset.append(seq)
                seq = []
        if len(seq) != 0:
            seq += [seq[-1]]*(tslen - len(seq))
            sampleset.append(seq)
    
    print 'loaded seq count =', len(sampleset)
    print 'loaded image count =', reduce(lambda x, y: x+len(y), sampleset, 0)
    return sampleset
    
def get_divided_seqid(arg, sampleset):
    ids = range(0, len(sampleset))
    np.random.shuffle(ids)
    validlen = int(len(ids)*arg['validportion'])
    validid = ids[0:validlen]
    trainid = ids[validlen:]
    return (trainid, validid)
    
def divide_seqid_tobatch(arg, idlist):
    batchsize = arg['batchsize']
    batch = []
    count = 0
    for i in range(0, len(idlist), batchsize):
        batch.append(idlist[i:i+batchsize])
    if len(batch)*batchsize < len(idlist):
        batch += [idlist[len(batch)*batchsize:]]
    return batch        
    
def init_param(arg):
    #def ortho_weight(ndim):
    #    W = numpy.random.randn(ndim, ndim)
    #    u, s, v = numpy.linalg.svd(W)
    #    return u.astype(config.floatX)

    # 各層のパラメータ(Wにはさらにi, f, c, o)
    # convLstm(cl): cl_WE, cl_WR, cl_WRPP, cl_WC, cl_b 
    # convReconstruct(cr): cr_W, cr_b
    # convToUpperLayer(cu): cu_W, cu_b
    
    param = {}
    factor = 0.1
    
    ks = arg['kernelsize']
    for i in range(0, arg['layercount']):
        ch, h, w = arg['layerinfo'][i]
        # {i, f, c, o}, filter, inputchannel, h, w
        param['cl_WE_' + str(i)] = floatX_array(np.random.randn(4, ch, 2*ch, ks, ks)*factor)
        param['cl_WR_' + str(i)] = floatX_array(np.random.randn(4, ch, ch, ks, ks)*factor)
        if i != arg['layercount'] -1:
            param['cl_WRPP_' + str(i)] = floatX_array(np.random.randn(4, ch, arg['layerinfo'][i+1][0], ks, ks)*factor)
        param['cl_WC_' + str(i)] = floatX_array(np.random.randn(4, ch, h, w)*factor)
        param['cl_b_' + str(i)] = floatX_array(np.zeros((4, ch, h, w)))
        
        # (filternum, inputchannelnum, h, w)
        param['cr_W_' + str(i)] = floatX_array(np.random.randn(ch, ch, ks, ks)*factor)
        param['cr_b_' + str(i)] = floatX_array(np.zeros((ch)))
        
        if i != arg['layercount'] -1:
            param['cu_W_' + str(i)] = floatX_array(np.random.randn(arg['layerinfo'][i+1][0], ch*2, ks, ks)*factor)
            param['cu_b_' + str(i)] = floatX_array(np.zeros((arg['layerinfo'][i+1][0])))
        
    return param

def load_model(filepath, param):
    p = np.load(filepath)
    for k, v in param.items():
        if k not in p: raise Exception('%s is not in the archive' % k)
        param[k] = p[k]

    return param

def init_theanoshared(param):
    theanoshared = {}
    for k, p in param.items(): theanoshared[k] = theano.shared(p, name=k)
    return theanoshared

def set_theanoshared(theanoshared, param):
    for k, v in param.items(): theanoshared[k].set_value(v)

def get_param(theanoshared):
    param = {}
    for k, v in theanoshared.items(): param[k] = v.get_value()
    return param

def adadelta(lr, theanoshared, grads, x, cost):
    # [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
    # Rate Method*, arXiv:1212.5701.
    
    zipped_grads = [theano.shared(p.get_value() * floatX_array(0.), 
        name='%s_grad' % k) for k, p in theanoshared.items()]
    running_up2 = [theano.shared(p.get_value() * floatX_array(0.),
        name='%s_rup2' % k) for k, p in theanoshared.items()]
    running_grads2 = [theano.shared(p.get_value() * floatX_array(0.),
        name='%s_rgrad2' % k) for k, p in theanoshared.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
    
    f_grad_shared = theano.function(
        [x], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg 
        for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
        for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(theanoshared.values(), updir)]

    f_update = theano.function(
        [lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update

def spad2d(x, plast=1, psecondlast=1):
    input_shape = x.shape
    output_shape = list(input_shape)
    output_shape[-1] += 2* plast
    output_shape[-2] += 2* psecondlast
    output = tensor.zeros(tuple(output_shape))
    indices = [slice(None)]*(len(output_shape) -2)
    indices += [slice(psecondlast, input_shape[-2] + psecondlast)]
    indices += [slice(plast, input_shape[-1] + plast)]
    return tensor.set_subtensor(output[tuple(indices)], x)

def ConvLSTM(theanoshared, batchsize, layerid, layerinfo, E, R, C, RPP, padw):
    #E: batchsize, 2, ch=numlayerch, h, w (conv)
    #R: batchsize, ch=numlayerch, h, w (conv)
    #C: batchsize, ch=numlayerch, h, w (elemwise)
    #b: batchsize, ch=numlayerch, h, w (elemwise)
    #RPP: batchsize, ch=2*numlayerch, h, w (conv)
    si = str(layerid)
    ch, h, w = layerinfo
    zi = tensor.nnet.conv.conv2d(spad2d(E, padw, padw), theanoshared['cl_WE_'+si][0])
    zi += tensor.nnet.conv.conv2d(spad2d(R, padw, padw), theanoshared['cl_WR_'+si][0])
    zi += C*(theanoshared['cl_WC_'+si][0])
    zi += theanoshared['cl_b_'+si][0]
    
    zf = tensor.nnet.conv.conv2d(spad2d(E, padw, padw), theanoshared['cl_WE_'+si][1])
    zf += tensor.nnet.conv.conv2d(spad2d(R, padw, padw), theanoshared['cl_WR_'+si][1])
    zf += theanoshared['cl_b_'+si][1]
    
    zc = tensor.nnet.conv.conv2d(spad2d(E, padw, padw), theanoshared['cl_WE_'+si][2])
    zc += tensor.nnet.conv.conv2d(spad2d(R, padw, padw), theanoshared['cl_WR_'+si][2])
    zc += C*(theanoshared['cl_WC_'+si][2])
    zc += theanoshared['cl_b_'+si][2]
    
    zo = tensor.nnet.conv.conv2d(spad2d(E, padw, padw), theanoshared['cl_WE_'+si][3])
    zo += tensor.nnet.conv.conv2d(spad2d(R, padw, padw), theanoshared['cl_WR_'+si][3])
    zo += C*(theanoshared['cl_WC_'+si][3])
    zo += theanoshared['cl_b_'+si][3]
    
    if RPP is not None:
        zi += tensor.nnet.conv.conv2d(spad2d(RPP, padw, padw), theanoshared['cl_WRPP_'+si][0])
        zf += tensor.nnet.conv.conv2d(spad2d(RPP, padw, padw), theanoshared['cl_WRPP_'+si][1])
        zc += tensor.nnet.conv.conv2d(spad2d(RPP, padw, padw), theanoshared['cl_WRPP_'+si][2])
        zo += tensor.nnet.conv.conv2d(spad2d(RPP, padw, padw), theanoshared['cl_WRPP_'+si][3])
    
    i = tensor.nnet.sigmoid(zi)
    f = tensor.nnet.sigmoid(zf)
    Cnext = f*C + tensor.tanh(zc)
    o = tensor.nnet.sigmoid(zo)
    Rnext = o * tensor.tanh(Cnext)
    return (Rnext, Cnext)
    
def Upsample(x):
    t = tensor.repeat(x, 2, axis=2)
    return tensor.repeat(t, 2, axis=3)

def build_model(arg, theanoshared):

    trng = RandomStreams(arg['randomseed'])
    li = arg['layerinfo']
    ll = arg['layercount']
    kernelsize = arg['kernelsize']
    timestep = arg['timesteplen']
    padw = (arg['kernelsize'] - 1) // 2
    off = 1e-8 if config.floatX != 'float16' else 1e-6
    
    # timestepがそろったseqをbatchsizeの分だけまとめて計算
    
    # input symbols
    # x: 1つのbatch 足は5つ [seqid, timestep, ch, w, h]
    x = tensor.TensorType(config.floatX, (False,)*5)('x')
    #usenoise = theano.shared(floatX_array(0.))
    
    # 各足の長さ
    batchsize = x.shape[0]
    
    # 各層のE, Rの領域を確保, 初期化
    print 'build model_phase_alloc'
    E = []
    R = []
    recA = []
    C = []
    for i in range(0, ll):
        ch, h, w = li[i]
        # Errorは+/-で2倍
        E.append(tensor.alloc(floatX_array(0.), batchsize, 2*ch, h, w))
        R.append(tensor.alloc(floatX_array(0.), batchsize, ch, h, w))
        recA.append(tensor.alloc(floatX_array(0.), batchsize, ch, h, w))
        C.append(tensor.alloc(floatX_array(0.), batchsize, ch, h, w))
    
    cost = 0.
    
    print 'build model_phase_timestep'
    # x_swapped: [timestep, seqid, ch, w, h]
    x_swapped = x.swapaxes(0, 1)
    for t in range(0, timestep):
        print 'handing timestep', 1+t, '/', timestep, ':',
        #
        for i in reversed(range(0, ll)):
            R[i], C[i] = ConvLSTM(theanoshared, batchsize, i, li[i], E[i], R[i], C[i], None if i == ll-1 else Upsample(R[i+1]), padw)
        print 'ConvLSTM',
        
        #
        for i in range(0, ll):
            tmp = tensor.nnet.conv.conv2d(spad2d(R[i], padw, padw), theanoshared['cr_W_'+str(i)])
            tmp += theanoshared['cr_b_'+str(i)][None, :, None, None]
            recA[i] = tensor.nnet.relu(tmp)
        recA[0] = tensor.minimum(recA[0], 1.0)
        print 'reconstruct_A',
        
        #
        A = x_swapped[t]
        for i in range(0, ll):
            ch, h, w = li[i]
            e1 = tensor.nnet.relu(recA[i] - A)
            e2 = tensor.nnet.relu(A - recA[i])
            E[i] = tensor.concatenate([e1, e2], axis=1)
            if i != ll-1:
                tmp = tensor.nnet.conv.conv2d(spad2d(E[i], padw, padw), theanoshared['cu_W_'+str(i)])
                tmp += theanoshared['cu_b_'+str(i)][None, :, None, None]
                A = tensor.signal.pool.pool_2d(tmp, (2, 2), ignore_border=True)
        print 'error_calc',
        
        #
        if t != 0: cost += E[0].mean()
        print 'cost'
    
    cost = cost / timestep
    f_image_err = theano.function([x], cost, name='f_image_err')
    
    return (x, cost, f_image_err)

def mean_error(f_image_err, sampleset, validbatch):
    err = []
    for batch in validbatch:
        x = floatX_array([sampleset[i] for i in batch])
        err.append(f_image_err(x))
    return np.array(err).mean()
    
def construct_model(arg):
    print 'construct_model'
    
    print 'parameter initializing'
    param = init_param(arg)
    if arg['premodelnpz']: param = load_model(arg['premodelnpz'], param)  
    theanoshared = init_theanoshared(param)

    print 'model building'
    (x, cost, f_image_err) = build_model(arg, theanoshared)

    print 'calc grad'
    grads = tensor.grad(cost, wrt=list(theanoshared.values()))
    #f_grad = theano.function([xs, masks], grads, name='f_grad')

    print 'calc optimizer'
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, theanoshared, grads, x, cost)
    
    return {'theanoshared':theanoshared, 'f_grad_shared':f_grad_shared, 'f_update':f_update, 'f_image_err':f_image_err}

def train_network(arg, model, sampleset):
    print 'train_network'
    
    # function compilation
    theanoshared = model['theanoshared']
    f_grad_shared = model['f_grad_shared']
    f_update = model['f_update']
    f_image_err = model['f_image_err']
    
    trainseqid, validseqid = get_divided_seqid(arg, sampleset)
    print '%d seqs for train, %d seqs for valid'%(len(trainseqid), len(validseqid))
    validbatch = divide_seqid_tobatch(arg, validseqid)
    print '%d valid batches made'%(len(validbatch))
    
    dispfreq = arg['dispfreq']
    savefreq = arg['savefreq']
    costsavefreq = arg['costsavefreq']
    validfreq = arg['validfreq']
    maxepochs = arg['maxepochs']
    batchsize = arg['batchsize']
    patience = arg['patience']
    lrate = arg['lrate']
    
    zippath = './' + arg['modelname'] + '.npz'
    pklpath = './' + arg['modelname'] + '.pklb'
    
    arg['history'] = []
    history = arg['history']
    arg['costhistory'] = []
    costhistory = arg['costhistory']
    bestparam = None
    updatecount = 0
    earlystop = False
    epochid = 0
    badcontinue = 0
    
    starttime = time.time()
    
    validerr = mean_error(f_image_err, sampleset, validbatch)
    history.append([updatecount, validerr])
    print u'Validation error', validerr
    
    # to make sure finalization be processed
    try:
        for epochid in xrange(0, maxepochs):
            print u'%d epoch starts'%epochid
            nsample = 0
            np.random.shuffle(trainseqid)
            trainbatch = divide_seqid_tobatch(arg, trainseqid)
            print '%d train batches made'%(len(trainbatch))
            
            for batch in trainbatch:
                updatecount += 1
                nsample += len(batch)
                x = floatX_array([sampleset[i] for i in batch])
                cost = f_grad_shared(x)
                f_update(lrate)
                
                if np.isnan(cost) or np.isinf(cost): Exception('bad cost exception')
                
                if updatecount % dispfreq == 0:
                    print u'Epoch', epochid, u'UpdateCount', updatecount, 'Cost' , cost
                    
                if updatecount % costsavefreq == 0:
                    costhistory.append([epochid, updatecount, cost])
                
                if updatecount % savefreq == 0:
                    print u'Saving...'
                    param = bestparam if bestparam != None else get_param(theanoshared)
                    np.savez(zippath, **param)
                    with open(pklpath, 'wb') as f: pickle.Pickler(f).dump(arg)
                    print u'Done'

                if updatecount % validfreq == 0:
                    validerr = mean_error(f_image_err, sampleset, validbatch)
                    history.append([updatecount, validerr])

                    if (bestparam == None or validerr <= np.array(history)[:,1].min()):
                        bestparam = get_param(theanoshared)
                        badcontinue = 0

                    print u'Validation error', validerr

                    if (len(history) > patience and validerr >= np.array(history)[:-patience,1].min()):
                        badcontinue += 1
                        if badcontinue > patience:
                            print u'Early Stop!'
                            earlystop = True
                            break
                
            print u'%d sample seen in %d epoch'%(nsample, epochid)
            if earlystop: break
            
    except KeyboardInterrupt:
        print("Training interupted")
    
    endtime = time.time()
    
    if bestparam == None: bestparam = get_param(theanoshared)
    np.savez(zippath, **bestparam)
    with open(pklpath, 'wb') as f: pickle.Pickler(f).dump(arg)
    
    print u'%d epochs with %f sec/epochs' %((epochid + 1), (endtime - starttime) / (float(epochid + 1)))
    print u'Training took %.1fs'%(endtime - starttime)
    
    return None
    
if __name__ == '__main__':
    arg = {}
    arg['dispfreq'] = 10
    arg['costsavefreq'] = 50
    arg['validfreq'] = 200
    arg['savefreq'] = 200
    arg['maxepochs'] = 500
    
    arg['imgw'] = 128
    arg['imgh'] = 72
    arg['imgch'] = 3
    arg['samplesetpath'] = ['./sample_anime']
    arg['timesteplen'] = 8
    arg['validportion'] = 0.1
    arg['validbatchsize'] = 2

    # レイヤー情報(下層から) (チャンネル数(フィルタ数), height, width)
    # 第0層は入力イメージと同一にする
    # 層が上がるごとにheight, widthは半減する(maxpoolのせい)
    # arg['layerinfo'] = [(3, 144, 256), (16, 72, 128), (32, 36, 64), (64, 18, 32), (128, 9, 16)]
    arg['layerinfo'] = [(3, 72, 128), (8, 36, 64), (32, 18, 32), (128, 9, 16)]
    arg['layercount'] = len(arg['layerinfo'])
    # すべてのconvのカーネルサイズ
    arg['kernelsize'] = 5
    
    arg['lrate'] = 0.0001
    arg['patience'] = 10
    arg['batchsize'] = 2
    arg['noisestd'] = 0.
    # arg['usedropout'] = True
    arg['premodelnpz'] = None
    arg['modelname'] = 'model_anime'
    arg['randomseed'] = 9973
    
    np.random.seed(arg['randomseed'])
    
    # sampleset: [batchid, ch1, ch2, ch3, ...]
    # 値は255で割って0～1のfloatXとしてある.
    sampleset = get_sampleset(arg)
    model = construct_model(arg)
    train_network(arg, model, sampleset)
    
    print 'all processes are ended'
