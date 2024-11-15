import subprocess

commad = ['python3', 'save_txt_video.py']  # window: python  linux: python3
with open('log/part2_bot_log.txt', 'w')as f:
    subprocess.run(commad, stdout=f, stderr=subprocess.STDOUT)

