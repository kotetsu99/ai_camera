#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os, sys

# 各データセットディレクトリパス定義
org_dir = "dataset/01-org/"
face_dir = "dataset/02-face/"

# OpenCV物体検出サイズ定義
cv_width, cv_height = 32, 32
#cv_width, cv_height = 200, 200
# 顔画像リサイズ幅定義
img_width, img_height = 64, 64

# 顔検出用カスケードxmlファイルパス定義
cascade_xml = "haarcascade_frontalface_alt.xml"


def main():
    # 人物リスト定義
    #name_list = os.listdir(org_dir)
    if len(sys.argv)==1:
        print('使用法: python 01-face_detection.py person1 person2 ... ')
        sys.exit()
    name_list = sys.argv[1:]
    print(name_list)

    # org配下の人物名ディレクトリ毎に顔検出し、faceディレクトリ配下に保存
    for name in name_list:
        # 人物画像ディレクトリ定義
        print(name)
        org_char_dir = org_dir + name + "/"
        print(org_char_dir)
        # 顔画像保存先ディレクトリ定義
        face_char_dir = face_dir + name + "/"
        print(face_char_dir)

        # 顔検出実行
        detect_face(org_char_dir, face_char_dir)


def detect_face(org_char_dir, face_char_dir):
    # 人物画像ファイルリストを取得
    image_list = os.listdir(org_char_dir)
    print(image_list)

    # 画像ファイル毎に顔検出
    for image_file in image_list:
        # 画像ファイル読み込み
        org_image = cv2.imread(org_char_dir + image_file)
        if org_image is None:
            print("Not open:",image_file)
            continue

        # グレースケール変換
        image_gs = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
        # 顔検出実行
        cascade = cv2.CascadeClassifier(cascade_xml)
        # 見落としを抑えるため、最初は低精度で検出。徐々に精度を上げていく。
        for i_mn in range(1, 4, 1):
            # 顔検出
            face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=i_mn, minSize=(cv_width, cv_height))

            # 顔が1つ以上検出された場合、64x64のサイズで取得
            if len(face_list) > 0:
                for i, rect in enumerate(face_list):
                    image = org_image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                    if image.shape[0] < cv_width or image.shape[1] < cv_height:
                        continue
                    face_image = cv2.resize(image,(img_width, img_height))
                    print(face_image.shape)
                    # 顔画像をファイルに保存
                    face_file_name = os.path.join(face_char_dir, "face_" + str(i_mn)  + "_"  + str(i) + "-" + image_file)
                    cv2.imwrite(str(face_file_name), face_image)
            # 顔が検出されなかった場合スキップして次の画像へ
            else:
                print("no face")
                continue


if __name__ == '__main__':
    main()
