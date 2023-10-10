# -*- coding: utf-8 -*-
"""InterFaceGAN

"""# Define Utility Functions"""

import os.path
import io
import IPython.display
import numpy as np
import cv2
import PIL.Image
import qrcode
import torch
from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.manipulator import linear_interpolate
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import gc
import torch.nn as nn
from stylegan_model import G_mapping
from stylegan_model import G_synthesis
from collections import OrderedDict
import torchvision
from torchvision import models 
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.optim as optim
import os
import keras.backend as K
import tensorflow as tf
import sys
import yaml
from argparse import ArgumentParser
from tqdm.auto import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from getpass import getpass
import urllib.request
import replicate
from Google import Create_Service
from googleapiclient.http import MediaFileUpload
import pandas as pd
import qrcode
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def build_generator(model_name):
  """Builds the generator by model name."""
  gan_type = MODEL_POOL[model_name]['gan_type']
  if gan_type == 'pggan':
    generator = PGGANGenerator(model_name)
  elif gan_type == 'stylegan':
    generator = StyleGANGenerator(model_name)
  return generator


def sample_codes(generator, num, latent_space_type='Z', seed=0):
  """Samples latent codes randomly."""
  np.random.seed(360)
  codes = generator.easy_sample(num)
  #print("code = ",codes)
  #print(type(codes))
  np.save('sample_latent.npy', latent.detach().cpu().numpy())
  #print("code.shape = ",codes.shape())
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
    codes = generator.get_value(generator.model.mapping(codes))
  return codes


def imshow(images, col, viz_size=256):
  """Shows images in one figure."""
  num, height, width, channels = images.shape
  assert num % col == 0
  row = num // col

  fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

  for idx, image in enumerate(images):
    i, j = divmod(idx, col)
    y = i * viz_size
    x = j * viz_size
    if height != viz_size or width != viz_size:
      image = cv2.resize(image, (viz_size, viz_size))
    fused_image[y:y + viz_size, x:x + viz_size] = image

  fused_image = np.asarray(fused_image, dtype=np.uint8)
  data = io.BytesIO()
  PIL.Image.fromarray(fused_image).save(data, 'jpeg')
  im_data = data.getvalue()
  disp = IPython.display.display(IPython.display.Image(im_data))
  return disp

def image_reader(image_path, resize=None):
    with open(image_path, "rb") as f: # 특정 경로에서 이미지 불러오기
        image = Image.open(f)
        image = image.convert("RGB") # RGB 색상 이미지로 사용
    # 미리 정해 놓은 해상도에 맞게 크기 변환
    if resize != None:
        image = image.resize((resize, resize))
    transform = transforms.Compose([
        transforms.ToTensor() # [0, 1] 사이의 값을 가지는 Tensor 형태로 변형
    ])
    image = transform(image)
    image = image.unsqueeze(0) # 배치(batch) 목적의 차원 추가 (N, C, H, W)
    return image


# torch.Tensor 형태의 이미지를 화면에 출력하는 함수
def imshow_tensor(tensor):
    # matplotlib는 CPU 기반이므로 CPU로 옮기기
    image = tensor.cpu().clone()
    # torch.Tensor에서 사용되는 배치 목적의 차원(dimension)을 제거
    image = image.squeeze(0)
    gray_scale = False # 흑백 이미지 여부
    if image.shape[0] == 1:
        gray_scale = True
    # PIL 객체로 변경
    image = transforms.ToPILImage()(image)
    # 이미지를 화면에 출력(matplotlib는 [0, 1] 사이의 값이라고 해도 정상적으로 출력)
    if gray_scale: # 흑백인 경우 흑백 색상으로 출력
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()

class FeatureExtractor(torch.nn.Module):
    def __init__(self, n_layers):
        super(FeatureExtractor, self).__init__()
        extractor = models.vgg16(pretrained=True).features

        # 각 레이어까지의 출력 값을 계산 (n_layers=[1, 3, 13, 20])
        index = 0
        self.layers = nn.ModuleList([])
        for i in range(len(n_layers)):
            # 해당 레이어까지의 출력 값을 내보낼 수 있도록 하기
            self.layers.append(torch.nn.Sequential())
            for j in range(index, n_layers[i] + 1):
                self.layers[i].add_module(str(j), extractor[j])
            index = n_layers[i] + 1

        # 모델을 학습할 필요는 없으므로 기울기 추적 중지
        for param in self.parameters():
            param.requires_grad = False

    # 각 레이어까지의 출력 값들을 리스트에 담아 반환
    def forward(self, x):
        result = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            result.append(x)

        return result

def loss_function(generated_image, target_image, feature_extractor):
    MSE = nn.MSELoss(reduction='mean')
    mse_loss = MSE(generated_image, target_image) # 손실(loss) 값 계산

    # VGG 네트워크의 입력은 256이므로, 크기를 256으로 바꾸는 업샘플링(upsampling)을 이용합니다.
    upsample2d = torch.nn.Upsample(scale_factor=256 / resolution, mode='bilinear')
    real_features = feature_extractor(upsample2d(target_image))
    generated_features = feature_extractor(upsample2d(generated_image))

    perceptual_loss = 0
    # 활성화 맵(activation map)의 개수만큼 반복하며
    for i in range(len(real_features)):
        perceptual_loss += MSE(real_features[i], generated_features[i]) # 손실(loss) 값 계산

    return mse_loss, perceptual_loss

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def age_predict(face_pic):
    err=0
    faces = face_cascade.detectMultiScale(face_pic,scaleFactor=1.11, minNeighbors=8)
    col = (255,255,0)
    image_size = 200
    age_ = []
    age_gap_=[]
    abs_age_gap_=[]
    for (x,y,w,h) in faces:
        img = pic[y:y + h, x:x + w]
        img = cv2.resize(img,(image_size,image_size))
        age_predict = age_model.predict(np.array(img).reshape(-1,image_size,image_size,3))
        age_.append(age_predict)
        age_gap_.append(int(age_predict)-int(cur_age))
        abs_age_gap_.append(abs(int(age_predict)-int(cur_age)))
        cv2.rectangle(pic,(x,y),(x+w,y+h),(0,225,0),3)
        #cv2.putText(pic,"Age:"+str(int(age_predict)),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,w*0.005,5,5) #col?? 일단 제거함
        cv2.putText(pic,"Age:"+str(int(age_predict)),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,w*0.005,col,5)

    print(age_)
    #plt.figure(figsize=(20,16))
    #plt.imshow(pic1)
    #plt.show()
    if len(age_)!=5:
        err=1
    plt.imsave("/home/shpark/result_group/age_predict_0609/result_"+str(cur_age)+"_?_"+str(imgNum)+"_"+str(predict_num)+".jpg", pic)
    print(age_gap_)
    sorted_age_=sorted(age_gap_)
    print(sorted_age_)
    abs_age_gap=min(abs_age_gap_)
    try:
        age_index=sorted_age_.index(abs_age_gap)
    except:
        age_index=listRightIndex(sorted_age_, -abs_age_gap)
    age_gap=sorted_age_[age_index]
    return age_index, age_gap, err

def gender_predict(face_pic):
    faces = face_cascade.detectMultiScale(face_pic,scaleFactor=1.11, minNeighbors=8)
    gender_=[]
    image_size = 200
    for (x,y,w,h) in faces:
        img = pic[y:y + h, x:x + w]
        img = cv2.resize(img,(image_size,image_size))
        gender_predict = gender_model.predict(np.array(img).reshape(-1,image_size,image_size,3))
        gender_.append(gender_predict)
        gend = np.round(gender_predict)
        if gend == 0:
            gender = 'Man'
            gp=0
            col = (255,255,0)
        else:
            gender = 'Woman'
            gp=1
            col = (203,12,255)
        cv2.rectangle(pic,(x,y),(x+w,y+h),(0,225,0),3)
        cv2.putText(pic,str(gender),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,w*0.005,col,5)
        K.clear_session()
    print(gender)
    plt.imsave("age_predict/result_gender.jpg", pic)
    print("age_predict/result_gender.jpg 저장 완료")
    return gp


def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment  # type: ignore (local file)
    from scipy.spatial import ConvexHull

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

#stylegan editting
from pyparsing.helpers import empty
import cv2
import os
import glob
#from google.colab.patches import cv2_imshow
#라이브러리 import
age_model_path = "/home/shpark/capston/model_0609/age_model.h5"
gender_model_path = "/home/shpark/capston/model_0609/gender_model.h5"
#age_model = load_model(age_model_path)
#gender_model = load_model(gender_model_path)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=270)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

age_model = load_model(age_model_path, compile=False)
gender_model = load_model(gender_model_path,compile=False)

#얼굴 검출을 위한 OPENCV Cascadeclassifier load
imgNum  = 11336
face=0
path = "/home/shpark/capston/korean2"
files = glob.glob(path + '/*')

for f in files: #실제로는 카메라를 계속 촬영하는 while문, 촬영 사진이 생성될 경우 try문 시행
    try:
        img = cv2.imread(f)
        filename = os.path.split(f)[1]
        print(filename)
        file = os.path.splitext(filename)[0]
        print(os.path.basename(file))
        #filename = os.path.splitext(filename)[0]
        #print(os.path.basename(filename))
        _,_,pic_gender, pic_age,_,_= file.split('_')

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        resolution = 1024
        weight_file = 'weights/karras2019stylegan-ffhq-1024x1024.pt'
        # StyleGAN을 구성하는 두 개의 네트워크 불러오기
        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('g_synthesis', G_synthesis(resolution=resolution))    
        ]))
        g_all.load_state_dict(torch.load(weight_file, map_location=device))
        g_all.eval()
        g_all.to(device)

        g_mapping, g_synthesis = g_all[0], g_all[1]

        del g_mapping
        """# Sample latent codes"""
        #@title { display-mode: "form", run: "auto" }
        latent = torch.zeros((1, 18, 512), requires_grad=True, device=device) # 업데이트할 latent vector 변수
        #print(latent)
        optimizer_latent = optim.Adam({latent}, lr=0.02, betas=(0.9, 0.999), eps=1e-8)

        # VGG perceptual loss를 위한 레이어 명시
        feature_extractor = FeatureExtractor(n_layers=[1, 3, 13, 20]).to(device)

        os.environ["REPLICATE_API_TOKEN"] = "r8_4dyM3TfgeRGQt3dAQIOiWV3peIHAAbj27QVPl"
        #past_age=input("사진의 나이를 입력해주세요") #사진의 나이가 0세 또는 90세가 넘을 경우 입력받지 않는 조건문 형성
        past_age=pic_age
        cur_age=int(pic_age)+25
        #cur_age=input("현재 나이를 입력해주세요")
        if cur_age>=100:
            print("100세 이상의 나이는 퀄리티의 문제로 인하여 출력하지 않습니다.")
            continue
        real_age_gap=int(cur_age)-int(past_age)
        if real_age_gap<5:
            print("원하는 나이와 현재의 나이 차이가 크지 않거나, 원하는 나이가 현재 나이보다 적으므로 출력하지 않습니다.")
            continue
        if pic_gender=='D' or pic_gender=='D2' or pic_gender=='M' or pic_gender=='GM':
            my_gender=1
        elif pic_gender=='S' or pic_gender=='S2' or pic_gender=='F'or pic_gender=='GF':
            my_gender=0
        else:
            print("여자와 남자를 제외한 다른 성별은 입력받지 않습니다.") #실제 시행 때는 입력이 0과 1이 아닐 경우
            continue
        #my_gender=input("성별을 입력해주세요(여자:1 / 남자:0)")
        if int(pic_age)<20:
            if int(my_gender)==0:
                if real_age_gap<10:
                    neutral="a chubby face"
                    target="a male face"
                    manipulation_strength= 0.2
                elif real_age_gap>=10 and real_age_gap<25:
                    neutral= "a male face"
                    target="a mature hairless male face"
                    manipulation_strength= 3
                elif real_age_gap>=25 and real_age_gap<50:
                    neutral= "a young face"
                    target="a old face with wrinkle"
                    manipulation_strength=  4
                elif real_age_gap>=50 and real_age_gap<80:
                    neutral= "a young face"
                    target="a old face with wrinkle"
                    manipulation_strength=  7
                if int(cur_age)>80:
                    neutral= "a young boy face"
                    target="an old man with wrinkle and white hair"
                    manipulation_strength= 8

            elif int(my_gender)==1:
                if real_age_gap<10:
                    neutral="a chubby face"
                    target="a female face"
                    manipulation_strength= 0.2
                if real_age_gap>=10 and real_age_gap<25:
                    neutral="a face"
                    target="mature female face"
                    manipulation_strength=  5
                elif real_age_gap>=25 and real_age_gap<50:
                    neutral= "a young face"
                    target="a old face with wrinkle"
                    manipulation_strength=  4
                elif real_age_gap>=50 and real_age_gap<80:
                    neutral= "a young face"
                    target="a old face with wrinkle"
                    manipulation_strength=  7
                if int(cur_age)>80:
                    neutral= "a young girl face"
                    target="an old lady  with wrinkle and white hair"
                    manipulation_strength= 8

            src=str(path)+"/"+str(filename)#input 입력 여기!!!

            output = replicate.run(
            "orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021",
            input={"input": open(src, "rb"), 
            "neutral":neutral,
            "target": target,
            "manipulation_strength":  manipulation_strength,
            "disentanglement_threshold":0.15}
            )
            image_url = output
            
            # URL에서 이미지 다운로드
            urllib.request.urlretrieve(image_url, "result.jpg")

            #동양화
            output = replicate.run(
            "orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021",
            input={"input": open("result.jpg", "rb"), 
            "target": "asian face",
            "manipulation_strength":  0.4,
            "disentanglement_threshold":0.15}
            )
            image_url = output

            # URL에서 이미지 다운로드
            urllib.request.urlretrieve(image_url, "result2.jpg")
            src = "result2.jpg"
            iteration=800
        else:
            iteration=1500
        name = 'barack_obama'
        image = image_reader(src, resize=1024)
        image = image.to(device)
        #print(type(latent_codes))

        #print(latent_codes.shape) #(1, 512)
        #save_image(g_synthesis(latent_codes), 'start.png')
        for i in range(iteration):
            optimizer_latent.zero_grad()

            # latent vector를 이용해 이미지 생성
            generated_image = g_synthesis(latent)
            generated_image = (generated_image + 1.0) / 2.0
            generated_image = generated_image.clamp(0, 1)

            # 손실(loss) 값 계산
            loss = 0
            mse_loss, perceptual_loss = loss_function(generated_image, image, feature_extractor)
            loss += mse_loss + perceptual_loss
            
            loss.backward() # 기울기(gradient) 계산
            optimizer_latent.step() # latent vector 업데이트

            # 주기적으로 손실(loss) 값 및 이미지 출력
            
            os.chdir("/home/shpark/capston/interfacegan/embedding/")
            if i == 0 or (i + 1) % 100 == 0:
                #print(f'[iter {i + 1}/{iteration}] loss = {loss.detach().cpu().numpy()}, saved_path = {name}_{i + 1}.png')
                save_image(generated_image, f'{name}_{i + 1}.png')
                #print(os.getcwd())
                os.chdir("/home/shpark/capston/interfacegan")
                np.save(f'{name}_latent.npy', latent.detach().cpu().numpy())

        print("[ 임베딩 결과 ]")
        print(type(generated_image))
        imshow_tensor(generated_image)

        # with torch.no_grad(): # GPU 용량 낭비를 줄이기 위해 기울기를 추적하지 않도록 하기
        # 바운더리(boundary) 불러오기
        boundary_name = 'age.npy'
        boundary_name_2 = 'stylegan_ffhq_gender_w_boundary.npy'
        boundary_glasses='stylegan_ffhq_eyeglasses_w_boundary.npy'
        boundary = np.load(f'boundaries/{boundary_name}')
        boundary2 = np.load(f'boundaries/{boundary_name_2}')
        boundary4=np.load(f'boundaries/{boundary_glasses}')
        # 이미지 임베딩 불러오기
        name = 'age'
        latent = np.load('barack_obama_latent.npy')

        boundary_images = []


        # 방향 벡터를 활용해 시맨틱 정보 변경하기
        #라이브러리 import
        
       

        #얼굴 검출을 위한 OPENCV Cascadeclassifier load
        face_cascade = cv2.CascadeClassifier("/home/shpark/capston/haarcascade_frontalface_default.xml")

        age_gap=100000
        age_index=0
        predict_num=1
        spectrum=3.0
        last=0
        number=0
        with torch.no_grad(): # GPU 용량 낭비를 줄이기 위해 기울기를 추적하지 않도록 하기
            while abs(age_gap)>5:
                if age_index==4:
                    spectrum=spectrum*4

                boundary_images=[]
                latent = np.load('barack_obama_latent.npy')
                boundary = boundary.reshape(1, 1, -1)
                boundary2 = boundary2.reshape(1, 1, -1)
                boundary4 = boundary4.reshape(1, 1, -1)
                linspace = np.linspace(last, last+spectrum, 5)
                linspace = linspace.reshape(-1, 1, 1)
                plus=True
                if predict_num==1:
                    w=latent +linspace* boundary
                    plus=True
                elif predict_num>1 and int(age_gap)<0:
                    w = latent +linspace* boundary
                    plus=True
                elif predict_num!=1 and int(age_gap)>0:
                    linspace = np.linspace(last-spectrum, last, 5)
                    linspace = linspace.reshape(-1, 1, 1)
                    w = latent + (linspace)* boundary
                    plus=False
                print("age_gap: "+ str(age_gap))
                w1 = torch.tensor(w).to(torch.float).to(device).clone().detach()
                image = g_synthesis(w1)
                image = (image + 1.0) / 2.0
                image = image.clamp(0, 1)
                boundary_images.append(image)

                grid_image = torchvision.utils.make_grid(torch.cat(boundary_images, dim=0), nrow=5)
                save_image(grid_image, f"{name}"+str(predict_num)+".png")
                
                pic = cv2.imreadpic = cv2.imread("/home/shpark/capston/interfacegan/age"+str(predict_num)+".png")
                age_index, age_gap, err=age_predict(pic)
                spectrum=spectrum/4

                if plus is True:
                    last=last+age_index*spectrum
                    print("last:"+str(last))
                else:
                    last=last-(5-age_index-1)*spectrum
                # if predict_num==1 and age_index==4:
                #     break
                if predict_num==3 and age_index==4 or err==1:
                    err=1
                    break
                print("완 1")

                #만일 age_predict에서 예측한 나이 행렬의 원소가 다섯 개가 아닐 경우: 스타일클립으로 노화해야 할지 고민 (age_gender_char_asian_result_64_1_11345.jpg 참고) 적용해야 할 듯


                predict_num=predict_num+1
                gc.collect()
                torch.cuda.empty_cache()

            if predict_num==3 and age_index==4 or err==1:
                if int(my_gender)==1:
                    if cur_age>90:
                        neutral= "a young girl face"
                        target="an old lady with wrinkle and white hair"
                        manipulation_strength= 10
                    elif cur_age>70 and cur_age<=90:
                        neutral= "a young girl face"
                        target="an old woman with wrinkle and white hair"
                        manipulation_strength= 7.5
                    elif cur_age>60 and cur_age<=70:
                        neutral= "a young face"
                        target="a old face with wrinkle"
                        manipulation_strength=  6.3
                    elif cur_age>50 and cur_age<=60:
                        neutral= "a young face"
                        target="a old face with wrinkle"
                        manipulation_strength=  4.5
                    elif cur_age>40 and cur_age<=50:
                        neutral="a face"
                        target="mature woman face"
                        manipulation_strength=  7.5
                    elif cur_age>30 and cur_age<=40:
                        neutral="a face"
                        target="mature woman face"
                        manipulation_strength=  5
                    elif cur_age>20 and cur_age<=30:
                        neutral="a face"
                        target="a woman face with makeup"
                        manipulation_strength=  8
                    elif cur_age>10 and cur_age<=20:
                        neutral="a face"
                        target="a teenage girl face"
                        manipulation_strength=  7
                    else: 
                        neutral="a face"
                        target="a baby face"
                        manipulation_strength=  7
                if int(my_gender)==0:
                    if cur_age>90:
                        neutral= "a young boy face"
                        target="an old man with wrinkle and white hair"
                        manipulation_strength= 10
                    elif cur_age>70 and cur_age<=90:
                        neutral= "a young boy face"
                        target="an old man with wrinkle and white hair"
                        manipulation_strength= 7.5
                    elif cur_age>60 and cur_age<=70:
                        neutral= "a young face"
                        target="an old face with wrinkle"
                        manipulation_strength=  6.3
                    elif cur_age>50 and cur_age<=60:
                        neutral= "a young face"
                        target="an old face with wrinkle"
                        manipulation_strength=  4.5
                    elif cur_age>40 and cur_age<=50:
                        neutral="a male face"
                        target="a mature hairless male face"
                        manipulation_strength=  7.5
                    elif cur_age>30 and cur_age<=40:
                        neutral="a male face"
                        target="mature 30 years old hairless male face"
                        manipulation_strength=  10
                    elif cur_age>20 and cur_age<=30:
                        neutral="a hairless face"
                        target="a hairless male face"
                        manipulation_strength=  8
                    elif cur_age>10 and cur_age<=20:
                        neutral="a face"
                        target="a teenage boy face"
                        manipulation_strength=  7
                    else: 
                        neutral="a face"
                        target="a baby face"
                        manipulation_strength=  7
                output = replicate.run(
                    "orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021",
                    input={"input": open(src, "rb"), #input 입력 여기!!!
                    "neutral":neutral,
                    "target": target,
                    "manipulation_strength":  manipulation_strength,
                    "disentanglement_threshold":0.15}
                )
                image_url = output

                # URL에서 이미지 다운로드
                urllib.request.urlretrieve(image_url, "age_final.png")
                urllib.request.urlretrieve(image_url, "age_gender_final.jpg")
                #     #동양화
                output = replicate.run(
                "orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021",
                input={"input": open("age_gender_final.jpg", "rb"), #input 입력 여기!!!
                "target": "asian face",
                "manipulation_strength":  1.0,
                "disentanglement_threshold":0.15}
                )
                image_url = output
                # URL에서 이미지 다운로드
                urllib.request.urlretrieve(image_url, "age_gender_asian_result.jpg")
                print("err=1")
            else:
                chunks=w1.chunk(5, dim=0)
                final_image=(g_synthesis(chunks[age_index])+1.0)/2.0
                save_image(final_image, f"{name}_final.png")
                    

                pic = cv2.imreadpic = cv2.imread("/home/shpark/capston/interfacegan/age_final.png")
                gp=gender_predict(pic)
                
                linspace = np.linspace(last,last, 1)
                linspace = linspace.reshape(-1, 1, 1)

                if int(gp)==int(my_gender):
                    if int(my_gender)==1:
                        w2 = latent +linspace*boundary-boundary2-boundary4
                    elif int(my_gender)==0: 
                        w2 = latent +linspace*boundary+boundary2-boundary4
                    #w2=latent+linspace*boundary+0.2*boundary_q
                    w2=latent+linspace*boundary#-0.05*boundary4+0.2*boundary_q
                elif int(my_gender)==1: #여자
                    #w2 = latent +linspace*boundary-0.2*boundary2-0.05*boundary_q
                    w2 = latent +linspace*boundary-2*boundary2-boundary4#0.2*boundary_q
                elif int(my_gender)==0: #남자
                    #w2 = latent +linspace*boundary+0.1*boundary2-0.05*boundary_q
                    w2 = latent +linspace*boundary+2*boundary2-boundary4#-0.05*boundary4+0.2*boundary_q

                w3 = torch.tensor(w2).to(torch.float).to(device)
                print(w3.shape)
                #plt.figure(figsize=(20,16))
                #plt.imshow(pic1)
                #plt.show()

                age_gender_final_image = g_synthesis(w3)
                save_image((age_gender_final_image+1.0)/2.0, f"{name}_gender_final.jpg")
                ################################################################################################################################################# 수정
                if pic_gender==0:
                    asian_strength=last*0.15
                else:
                    asian_strength=last*0.13
                ################################################################################################################################################
                #     #동양화
                output = replicate.run(
                "orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021",
                input={"input": open("age_gender_final.jpg", "rb"), #input 입력 여기!!!
                "target": "asian face",
                "manipulation_strength":  asian_strength,
                "disentanglement_threshold":0.15}
                )
                image_url = output
                # URL에서 이미지 다운로드
                urllib.request.urlretrieve(image_url, "age_gender_asian_result.jpg")
                print("err=0")
            print("age_gender_final.png 저장 완료")
        ch=7 #일괄적으로 안경 제거로 코드 돌림.
        #특징 추가 필요 (장발, 단발, 흰머리, 검정머리, 밝은 피부, 안경 유무, 쌍꺼풀 제거)
        eyelid=0
        if ch==0: # 안경
            neutral=" a face"
            target= "a face with glasses"
            manipulation_strength=4.0
        elif ch==1: # 안경 제거
            neutral="a face with glasses"
            target= "a face"
            manipulation_strength=2.0
        elif ch==2: # 금발
            neutral="a face with black hair"
            target="a face with blonde hair"
            manipulation_strength=6.0
        elif ch==3: # 흰 머리
            neutral="a face with black hair"
            target="a face with white hair"
            manipulation_strength=4.0
        elif ch==4: # 검정 머리
            neutral="a face with white hair"
            target="a face with black hair"
            manipulation_strength=4.0
        elif ch==5: # 단발
            neutral="a face with long hair"
            target="a face with short hair"
            manipulation_strength=2.0
        elif ch==6: # 장발
            neutral="a face with short hair"
            target="a face with long hair"
            manipulation_strength=2.0
        elif ch==7: # 밝은 피부
            neutral="a face with dark skin"
            target="a face with bright skin"
            manipulation_strength=5.0
        elif ch==8: # 쌍꺼풀 추가
            neutral="asian face"
            target="a face"
            manipulation_strength=1.0
            eyelid=1
        elif ch==9: # 쌍꺼풀 제거
            neutral="a face"
            target="asian face"
            manipulation_strength=0.8
            eyelid=1
        elif ch==10:
            neutral="a face"
            target="asian face"
            if err==1:
                manipulation_strength=0.9
            else:
                manipulation_strength=0.8
            eyelid=1
        else:
            print("없는 특징을 입력받았습니다. 다시 입력해주세요.")

        output = replicate.run(
        "orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021",
        input={"input": open("age_gender_asian_result.jpg", "rb"), #input 입력 여기!!!
        "neutral":neutral,
        "target": target,
        "manipulation_strength":  manipulation_strength,
        "disentanglement_threshold":0.15}
        )
        image_url = output

        # URL에서 이미지 다운로드
        urllib.request.urlretrieve(image_url, "age_gender_char_result.jpg")

        print("특징 제어 완료")
        #################################################################################### 수정
        if eyelid==0:
            if err==1:
                manipulation_strength=0.85
            elif err==0 and ch==1:
                manipulation_strength=1.1 
            else:
                manipulation_strength=0.8
         #############################################################################################       
            #동양화
            output = replicate.run(
            "orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021",
            input={"input": open("age_gender_char_result.jpg", "rb"), #input 입력 여기!!!
            "target": "asian face",
            "manipulation_strength":  manipulation_strength,
            "disentanglement_threshold":0.15}
            )
            image_url = output

            # URL에서 이미지 다운로드
            urllib.request.urlretrieve(image_url, "age_gender_char_asian_result.jpg")
        elif eyelid==1:
            urllib.request.urlretrieve(image_url, "age_gender_char_asian_result.jpg")
        print("최종 사진 저장")

        img1=cv2.imread(str(path)+"/"+str(filename))
        img2=cv2.imread("result.jpg")
        img3=cv2.imread("result2.jpg")
        img4=cv2.imread("age_gender_final.jpg")
        img5=cv2.imread("age_gender_asian_result.jpg")
        img6=cv2.imread("age_gender_char_result.jpg")
        img7=cv2.imread("age_gender_char_asian_result.jpg")
        img1=cv2.resize(img1, (640,640))
        img2=cv2.resize(img2, (640,640))
        img3=cv2.resize(img3, (640,640))
        img4=cv2.resize(img4, (640,640))
        img5=cv2.resize(img5, (640,640))
        img6=cv2.resize(img6, (640,640))
        img7=cv2.resize(img6, (640,640))
        addv=np.hstack((img1, img2))
        addv=np.hstack((addv, img3))
        addv=np.hstack((addv, img4))
        addv=np.hstack((addv, img5))
        addv=np.hstack((addv, img6))
        addv=np.hstack((addv, img7))
        cv2.imwrite("/home/shpark/result_group/output_sa_asa/"+str(cur_age)+"_"+str(my_gender)+"_"+str(imgNum)+"_err="+str(err)+".jpg", addv)
        #"/home/shpark/capston/output/age_gender_char_asian_result"+"_"+str(cur_age)+"_0_"+str(imgNum)+".jpg")
        imgNum =imgNum+ 1
        print(imgNum)
        if sys.version_info[0] < 3:
            raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
        if __name__ == "__main__":
            parser = ArgumentParser()
            parser.add_argument("--config", default='config/vox-adv-256.yaml', help="path to config")
            parser.add_argument("--checkpoint", default='checkpoint/vox-adv-cpk.pth.tar', help="path to checkpoint to restore")

            parser.add_argument("--source_image", default='/home/shpark/capston/interfacegan/age_gender_char_asian_result.jpg', help="path to source image")
            parser.add_argument("--driving_video", default='/home/shpark/capston/interfacegan/smile.mp4', help="path to driving video")
            parser.add_argument("--result_video", default='/home/shpark/capston/interfacegan/deepfake_video/result.mp4', help="path to output")

            parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
            parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

            parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                                help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

            parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")

            parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

            parser.add_argument("--audio", dest="audio", action="store_true", help="copy audio to output from the driving video" )

            parser.set_defaults(relative=False)
            parser.set_defaults(adapt_scale=False)
            parser.set_defaults(audio_on=False)

            opt = parser.parse_args()

            source_image = imageio.imread(opt.source_image)
            reader = imageio.get_reader(opt.driving_video)
            fps = reader.get_meta_data()['fps']
            driving_video = []
            try:
                for im in reader:
                    driving_video.append(im)
            except RuntimeError:
                pass
            reader.close()

            source_image = resize(source_image, (256, 256))[..., :3]
            driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
            generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

            if opt.find_best_frame or opt.best_frame is not None:
                i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
                print ("Best frame: " + str(i))
                driving_forward = driving_video[i:]
                driving_backward = driving_video[:(i+1)][::-1]
                predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
                predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
                predictions = predictions_backward[::-1] + predictions_forward[1:]
            else:
                predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
            imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

            if opt.audio:
                with NamedTemporaryFile(suffix='.' + splitext(opt.result_video)[1]) as output:
                    ffmpeg.output(ffmpeg.input(opt.result_video).video, ffmpeg.input(opt.driving_video).audio, output.name, c='copy').run()
                    with open(opt.result_video, 'wb') as result:
                        copyfileobj(output, result)

        try :
            import argparse
            flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
        except ImportError:
            flags = None

        store = file.Storage('storage.json')
        creds = store.get()

        service = build('drive', 'v3', http=creds.authorize(Http()))

        folder_id="1LskTzgN6WzcXQHItYgCYnNQ-gR97-0RS"

        request_body = {'name':"result", 'parents': [folder_id], 'uploadType':'mp4'}
        media = MediaFileUpload('/home/shpark/capston/interfacegan/deepfake_video/result.mp4',mimetype='video/mp4')
        file_info = service.files().create(body=request_body, media_body=media, fields='id,webViewLink').execute()
        if file_info:
            print("file upload success!")

        CLIENT_SECRET_FILE='client_secret_drive.json'
        API_NAME='drive'
        API_VERSION='v3'
        SCOPES=['https://www.googleapis.com/auth/drive']

        service=Create_Service(CLIENT_SECRET_FILE,API_NAME,API_VERSION,SCOPES)

        folder_id="1LskTzgN6WzcXQHItYgCYnNQ-gR97-0RS"

        query=f"parents='{folder_id}'"
        response=service.files().list(q=query).execute()

        files=response.get('files')
        file_id = files[0]['id']

        request_body={
            'role': 'reader',
            'type': 'anyone'
        }

        response_permission=service.permissions().create(
            fileId=file_id,
            body=request_body
        ).execute()
        print(response_permission)

        response_share_link = service.files().get(
            fileId=file_id,
            fields='webViewLink'
        ).execute()

        print(response_share_link)
        url = response_share_link['webViewLink']
        print(url)

        qr_img = qrcode.make(url)
        qr_img.save('qr_code/qr_test.png')
    ## QRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRr
    except:
        pass