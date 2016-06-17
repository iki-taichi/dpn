# dpn
deep predicitve coding networks demo

Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning
William Lotter, Gabriel Kreiman, David Cox
[https://arxiv.org/abs/1605.08104](https://arxiv.org/abs/1605.08104)

をとりあえず動かしてみようというサンプルです。
KerasやChainerでのモデリングは諦めて、theanoで実装しました。
かなり汚いです。

注
まだ、学習・予想という一連の動作が(エラーなく)動くようになったという段階で
学習がうまくいった例はありませんが、何か参考になれば幸いです

依存
theano, PIL

dpn_learn.py
* 画像を学習させる。
* 学習サンプルはディレクトリにまとめて番号を付けて入れておく(0.png, 1.png, 2.pngなど)
* 数値順にソートしてtimestepごとにわけseqenceとして使います。
* その他の設定は"if __name__ == '__main__':"以降を参照
* modelの構築にもかなり時間がかかるので気長に待つ必要があります。

dpn_eval.py
* loadで学習したモデルを読み込む。
* applyimg(imagepath or matrixdata)で画像を作用させる。
* matrixdataは3次元のデータで[channel, height, width]です。
* 各画素値は0.0-1.0に変換しておく必要があります。

以上
