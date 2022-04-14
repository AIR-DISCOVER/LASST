import os
import numpy as np

res = []

def getalltxt(filepath):
    if not os.path.isdir(filepath):
        return

    allfiles = os.listdir(filepath)
    for eachfile in allfiles:
        if eachfile.split(".")[-1] == "txt":
            res.append(os.path.join(filepath, eachfile))
        elif eachfile != "hsv_02" and eachfile != "base" and eachfile != "text2mesh_angel":
            newpath = os.path.join(filepath, eachfile)
            getalltxt(newpath)

txt_path = getalltxt("/home/jinbu/text2mesh/results/batch/2022-04-11_all_all")

print(res)

stylized_num = 0

all_iter = 0
ave_time = 0
ave_prepro_time = 0
ave_text_loss = 0
ave_norm_text_loss = 0
ave_hsv_loss = 0
ave_rgb_loss = 0



# print(res)
for path in res:
    with open(path, "r") as f:
        scores = f.read().split('\n')[1:]
        for score in scores:
            try:
                label           = np.float32(score[1:-1].split(',')[0])
                iter            = np.float32(score[1:-1].split(',')[1])
                prepro_time     = np.float32(score[1:-1].split(',')[2])
                all_tarin_time  = np.float32(score[1:-1].split(',')[3])
                text_loss       = np.float32(score[1:-1].split(',')[4])
                norm_text_loss  = np.float32(score[1:-1].split(',')[5])
                hsv_loss        = np.float32(score[1:-1].split(',')[6])
                rgb_loss        = np.float32(score[1:-1].split(',')[7])

                stylized_num += 1
                all_iter += iter
                ave_prepro_time += prepro_time
                ave_time += all_tarin_time
                ave_text_loss += text_loss
                ave_norm_text_loss += norm_text_loss
                ave_hsv_loss += hsv_loss
                ave_rgb_loss += rgb_loss
            
            except:
                pass

ave_time = ave_time/stylized_num
ave_prepro_time = ave_prepro_time/stylized_num
ave_text_loss = ave_text_loss/stylized_num
ave_norm_text_loss = ave_norm_text_loss/stylized_num
hsv_loss = hsv_loss/stylized_num
rgb_loss = rgb_loss/stylized_num

print(f"stylized_num:{stylized_num}")
print(f"ave_prepro_time:{ave_prepro_time}")
print(f"ave_time:{ave_time}")
print(f"ave_text_loss:{ave_text_loss}")
print(f"ave_norm_text_loss:{ave_norm_text_loss}")
print(f"hsv_loss:{hsv_loss}")
print(f"rgb_loss:{rgb_loss}")
