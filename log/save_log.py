import subprocess

commad = ['python', './save_txt_video.py']  # window: python  linux: python3
with open('./log/1.txt', 'w')as f:
    subprocess.run(commad, stdout=f, stderr=subprocess.STDOUT)

