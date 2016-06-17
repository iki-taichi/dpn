# dpn
deep predicitve coding networks demo ¥n
Taichi Iki 2016-06-17

Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning ¥n
William Lotter, Gabriel Kreiman, David Cox ¥n
[https://arxiv.org/abs/1605.08104](https://arxiv.org/abs/1605.08104)

をとりあえず動かしてみようというサンプルです。 ¥n
KerasやChainerでのモデリングは諦めて、theanoで実装しました。 ¥n
かなり汚いです。

注
まだ学習・予想という一連の動作が(エラーなく)動くようになったという段階で ¥n
学習がうまくいった例はありませんが、何か話題や参考になれば幸いです。 ¥n
また、うまくい学習しない理由をわかる方がいましたらぜひ連絡をください。 ¥n

## 依存
theano, PIL

## dpn_learn.py
* 画像を学習させる。
* 学習サンプルはディレクトリ(sample)にまとめて番号を付けて入れておく(0.png, 1.png, 2.pngなど)
* 数値順にソートしてtimestepごとにわけseqenceとして使います。
* その他の設定は"if __name__ == '__main__':"以降を参照
* modelの構築にもかなり時間がかかるので気長に待つ必要があります。
* 学習が進むと ****.pkl, ****.npzができます。これらが学習したモデルになります。

## dpn_eval.py
* loadで学習したモデルを読み込む。
* applyimg(imagepath or matrixdata)で画像を作用させる。
* matrixdataは3次元のデータで[channel, height, width]です。
* 各画素値は0.0-1.0に変換しておく必要があります。

以上
