import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt

def calculate(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    return [width, height]

def rescale_frame(frame, percent=75):
    width, height = calculate(frame, percent)
    #print("{} {}".format(width, height))
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def width_height(nome_video):
    cap = cv2.VideoCapture(nome_video)
    ret, frame = cap.read()
    width_out, height_out = calculate(frame, 50)
    cap.release()
    return [width_out, height_out]

def tratar_video(nome_video):
    video_dir = "videos_editados"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        
    cap = cv2.VideoCapture(nome_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width_out, height_out = width_height(nome_video)
    print("{} {}".format(width_out, height_out))
    old_name = nome_video.split('.')[0].split('\\')[1]
    new_name = video_dir+"\\"+old_name+"_editado.mp4"
    video_num_ref = int(old_name.split("_")[1])
    if video_num_ref != 360:
        out = cv2.VideoWriter(new_name, fourcc, 120.0, (width_out,height_out))
    else:
        out = cv2.VideoWriter(new_name, fourcc, 30.0, (width_out,height_out))
    
    taxa_atual_frame_rec = 1
    taxa_atual_frame_text = 7
    frames = 0
    while True:
        ret, frame = cap.read()
        frames += 1
        
        if ret == True:
            frame = cv2.transpose(frame)
            frame = cv2.transpose(frame)
            frame = rescale_frame(frame, percent=50)
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            cv2.putText(frame, "By: Vinicius G. Barros", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Attempts/Tentativas: {}".format(video_num_ref), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def edit_videos():
    myvideolist = [f for f in glob.glob("videos_edicao/*.mp4")]
    for video in myvideolist:
        tratar_video(video)

def create_black_image():
    #A customização do frame foi feita no site https://pixlr.com/x/
    #Font: Legendum
    #Size: 30 e 20
    blank_image = np.zeros(shape=[320, 480, 3], dtype=np.uint8)
    cv2.imwrite( "videos_editados/black_image.jpg", blank_image);
    cv2.imshow("Black Blank", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gerar_video_inicial():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    img_array = []
    for filename in glob.glob('*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        for _ in range(30):
            img_array.append(img)

    out = cv2.VideoWriter('intermediate_image_2.mp4', fourcc, 15.0, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

#SpaceX - https://www.youtube.com/watch?v=l5I8jaMsHYk
gerar_video_inicial()