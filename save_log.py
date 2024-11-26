import subprocess

commad = ['python3', 'save_txt_video.py']  # window: python  linux: python3
with open('./log/aiwei_2_reid_dist_224_log.txt', 'w')as f:
    subprocess.run(commad, stdout=f, stderr=subprocess.STDOUT)

