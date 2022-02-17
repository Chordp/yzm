from captcha.image import ImageCaptcha
import random
import time
captcha_lable = list('0123456789abcdefghijklmnopqrstuvwxyz')
captcha_size = 5
random.seed(time.time())
if __name__ == '__main__':
    for _ in range(10000):
        image = ImageCaptcha()
        captcha_text = ''.join(random.sample(captcha_lable, captcha_size))
        captcha_path = './data/train/{}_{}.png'.format(captcha_text,int(time.time()*1000))
        image.write(captcha_text,captcha_path)
