#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import numpy as np
import cv2
import picamera
import picamera.array
import os, sys
import time

# プログラム実行制限時間(分)
time_limit = 30
# 注文者本人の名前
person = 'person'

# OpenCV物体検出サイズ定義
cv_width, cv_height = 100, 100
# OpenCV物体検出閾値
minN = 4
# 顔画像サイズ定義
img_width, img_height = 64, 64
# 顔検出用カスケードxmlファイルパス定義
cascade_xml = "haarcascade_frontalface_alt.xml"

# 学習用データセットのディレクトリパス
train_data_dir = 'dataset/02-face'
# データセットのサブディレクトリ名（クラス名）を取得
classes = os.listdir(train_data_dir)


def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    print 'クラス名リスト = ', classes

    # 学習済ファイルの確認
    if len(sys.argv)==1:
        print('使用法: python ～.py 学習済ファイル名.h5')
        sys.exit()
    savefile = sys.argv[1]
    # モデルのロード
    model = keras.models.load_model(savefile)

    print('認識を開始')

    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            # カメラの解像度を320x240にセット
            #camera.resolution = (320, 240)
            camera.resolution = (480, 320)
            # カメラのフレームレートを15fpsにセット
            camera.framerate = 15
            # ホワイトバランスをfluorescent(蛍光灯)モードにセット
            camera.awb_mode = 'fluorescent'

            # 本人認識フラグ
            person_flg = False

            # 時間計測開始
            start_time = time.time()
            process_time = 0

            # 制限時間まで顔認識実行
            while process_time < time_limit :
                # stream.arrayにBGRの順で映像データを格納
                camera.capture(stream, 'bgr', use_video_port=True)

                # 顔認識
                image, person_flg = detect_face(stream.array, model, person_flg)
                # カメラ映像をウインドウに表示
                cv2.imshow('frame', image)

                # 'q'を入力でアプリケーション終了
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                # streamをリセット
                stream.seek(0)
                stream.truncate()

                # 経過時間(分)計算
                process_time = (time.time() - start_time) / 60
                #print('process_time = ', process_time, '[min]')

            cv2.destroyAllWindows()


def detect_face(image, model, person_flg):
    # グレースケール画像に変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_xml)

    # 顔検出の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=minN,minSize=(cv_width, cv_height))

    # 顔が1つ以上検出された場合
    if len(face_list) > 0:
        for rect in face_list:
            # 顔画像を生成
            face_img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if face_img.shape[0] < cv_width or face_img.shape[1] < cv_height:
                #print("too small")
                continue
            # 顔画像とサイズを定義
            face_img = cv2.resize(face_img, (img_width, img_height))

            # Keras向けにBGR->RGB変換、float型変換
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            # 顔画像をAIに認識
            name = predict_who(face_img, model)
            print(name)
            # 顔近傍に矩形描画
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness = 3)
            # AIの認識結果(人物名)を元画像に矩形付きで表示
            x, y, width, height = rect
            cv2.putText(image, name, (x, y + height + 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255) ,2)
            # 画像保存
            #cv2.imwrite(name + '.jpg', image)
            if name == person :
                person_flg = True
    return image, person_flg


def predict_who(x, model):
    # 画像データをテンソル整形
    x = np.expand_dims(x, axis=0)
    # 学習時に正規化してるので、ここでも正規化
    x = x / 255
    pred = model.predict(x)[0]

    # 予測確率が高いトップn個を出力
    top = 1
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], 1-pred[i]) for i in top_indices]
    print(result)
    print('=======================================')

    if result[0][1] > 0.5:
        name = classes[0]
    else:
        name = classes[1]

    # 1番予測確率が高い人物名を返す
    return name


if __name__ == '__main__':
    main()
