from django.shortcuts import render
from django.http import JsonResponse, HttpResponse

from music.nn.prediction import get_music
from music.nn.prediction import get_predict_music, net_music_list, net_recommend_genre, net_predict_music

import random
import os
from django.conf import settings


# import cv2
# import numpy as np
#
# # init_url = None
# # init_label = None
# # init_index = None
#
#
# ################################################
# BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
# SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 75
#
# SVM_SCORE_THRESHOLD = 1.8
# NMS_OVERLAP_THRESHOLD = 0.15
#
# sift = cv2.xfeatures2d.SIFT_create()
#
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = {}
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
#
# bow_kmeans_trainer = cv2.BOWKMeansTrainer(20)
# bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)
#
# def get_paths(i):
#
#     mouse_path = os.path.abspath('./music/static/svm/Train/mouse/mouse%d.jpg' % (i+1))
#     keyboard_path = os.path.abspath('./music/static/svm/Train/keyboard/keyboard%d.jpg' % (i + 1))
#     host_path = os.path.abspath('./music/static/svm/Train/host/host%d.jpg' % (i + 1))
#
#     return mouse_path, keyboard_path, host_path
#
# def add_sample(path):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     keypoints, descriptors = sift.detectAndCompute(img, None)
#     if descriptors is not None:
#         bow_kmeans_trainer.add(descriptors)
#
# for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
#     mouse_path, keyboard_path, host_path = get_paths(i)
#     add_sample(mouse_path)
#     add_sample(keyboard_path)
#     add_sample(host_path)
#
#
#
# voc = bow_kmeans_trainer.cluster()
# bow_extractor.setVocabulary(voc)
# def extract_bow_descriptors(img):
#     features = sift.detect(img)
#     return bow_extractor.compute(img, features)
#
# ################################################


def index(request):
    music_init_label, music_init_url, music_init_index = get_music()
    # global init_url
    # global init_label
    # global init_index
    image = f'./static/background/background_{random.randint(0, 7)}.jpg'
    init_index = music_init_index
    init_label = music_init_label
    init_url = music_init_url.replace("/music",'')
    print(f'index: {init_url}')
    content = [{"title": f'init: {init_label}',
                "artist": "Josh Yu",
                "mp3": init_url,
                "poster": image,
                "init_index":init_index},]

    return render(request, 'index.html', {"content": content})


# Related to index.html: Return JSON
def json_index(request):
    music_init_label, music_init_url, music_init_index = get_music()
    image = f'./static/background/background_{random.randint(0, 7)}.jpg'
    init_index = music_init_index
    init_index = music_init_index
    init_label = music_init_label
    init_url = music_init_url.replace("/music", '')
    print(f'index: {init_url}')
    content = [{"title": f'init: {init_label}',
                "artist": "Josh Yu",
                "mp3": init_url,
                "poster": image,
                "init_index": int(init_index)}, ]

    return JsonResponse(content, safe=False)

# Related to index.html: Return prediction information
def predicting(request):
    get_result = request.POST
    print(request.POST)
    image = f'./static/background/background_{random.randint(0, 7)}.jpg'
    max_music_url, music_init_index, real_label_index = get_predict_music(int(get_result["init_index"]))
    max_url = max_music_url.replace("/music",'')
    content = [{"title": f'{get_result["title"]}',
                "artist": f'predict: {real_label_index}',
                "mp3": max_url,
                "oga": "http://www.jplayer.org/audio/ogg/Miaow-07-Bubble.ogg",
                "poster": image,
                "init_index": get_result["init_index"]}]
    return JsonResponse(content, safe=False)

# Related to index.html: Return initially loaded song information
def previous(request):
    get_result = request.POST
    content = [{"title": get_result["title"],
                "artist": "The Starting Point",
                "mp3": get_result["mp3"],
                "poster": "https://file.fishei.cn/wallhaven-g89p2.png",
                "init_index": get_result["init_index"]},
               ]
    return JsonResponse(content, safe=False)


# Related to project.html: Return 5 randomly recommended songs of the same genre
def get_recommend_genre(request):
    music_list = net_recommend_genre()
    return JsonResponse({"musicList": music_list, "getId": 5190711437}, safe=False)

# Return project.html page
def project(request):
    return render(request, 'project.html')

# Related to project.html: Randomly recommend 5 songs
def get_music_list(request):
    music_list = net_music_list()
    return JsonResponse({"musicList": music_list, "getId": 5177783391}, safe=False)

# Related to project.html: Recommend songs based on liked song
def get_music_recommend(request):
    get_result = request.GET["musicIndex"]
    if len(get_result) == 0:
        get_result = random.sample(range(0, 10), 1)[0]
    print(get_result)
    music_list = net_predict_music(int(get_result))
    return JsonResponse({"musicList": music_list, "getId": 5190711435}, safe=False)
