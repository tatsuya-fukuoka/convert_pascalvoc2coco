## 1. 概要
* アノテーションツール[VOTT](https://github.com/Microsoft/VoTT/releases)で出力したPascal VOC形式のXMLファイルをCOCO形式のJSONファイルへ変換するコード
* VOTTで定義されたtrainとvalに分割を考慮して、COCO形式のデータセットに変換する。

## 2. 引用
* 今回は以下のGithubの内容を改変して作成しました。コア部分はこちらのコードを使用させていただきました。
* https://github.com/Kazuhito00/convert_voc_to_coco

## 3. 環境構築
### 3.1 pip install 
* tqdm
* natsort
```bash
pip install tqdm natsort
```

### 3.2 実行時のファイル構成
* convert_vott_voc2coco.pyのファイル構成は以下のようにします。
* 以下以外でも動くかもしれませんが、動作確認していません。
```txt
├─convert_vott_voc2coco.py
│  
└─{you setting name}-PascalVOC-export
    ├─Annotations
    ├─ImageSets
    │  └─Main
    └─JPEGImages
```
* Annotationsにはxmlファイル（@@@.xml...）、ImageSets/Mainにはtxtファイル(@@@_train.txt, @@@_val.txt)、JPEGImagesには画像ファイル（@@@.jpg...）を格納

## 4. 実行
* 以下のコマンドで実行します。
```bash
python convert_vott_voc2coco.py {yo setting name}-PascalVOC-export

# ex)
python convert_vott_voc2coco.py yolox_test-PascalVOC-export
```
* 実行後、以下のフォルダが生成されます。{YYYYMMDD_HHMMSS}は日付・時刻です。
* annotationsフォルダに変換後のtrain、valのjsonファイル（instances_train2017.json、instances_val2017.json）、train2017フォルダにtrain画像、val2017フォルダにval画像が保存されます。
```txt
─{YYYYMMDD_HHMMSS}_COCO_format
   ├─annotations
   |  ├─instances_train2017.json
   |  └─instances_val2017.json
   ├─train2017
   |  └─@@@.jpg
   └─val2017
      └─@@@.jpg
```
## 5. 活用
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)のdatasetsに格納して、学習に活用することができます。

## 6. 紹介
ブログ：https://chantastu.hatenablog.com/archive
Youtube：https://youtube.com/@chantatsu
