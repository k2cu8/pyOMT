import os
import os.path
import scipy.io as sio
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms


IMG_EXTENSIONS = ['.png', '.jpg']

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def load_im(path, class_to_idx):
    images = []
    dir = os.path.expanduser(path)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

if __name__ == '__main__':
    root = '/media/yang/data/research2018/data/CelebA_crop_resize_64'
    out_dir = 'data/'

    classes, class_to_idx = find_classes(root)
    imgs = load_im(root, class_to_idx)
    print('Found {} images'.format(len(imgs)))
    if len(imgs) == 0:
        quit()
    im0 = np.array(Image.open(imgs[0][0]))
    w,h,c = im0.shape
    print('Image has size {}x{}x{}'.format(w,h,c))
    im_stack = np.empty([len(imgs),w*h*c], dtype=np.float32)
    i=0
    for im_file, class_id in imgs:
        im_pil = Image.open(im_file)
        im_tensor = transforms.ToTensor()(im_pil)
        im_tensor = im_tensor.view(1,-1)
        im_stack[i,:] = im_tensor.numpy()
        i = i+1
    #sio.savemat(os.path.join(out_dir,'im_stack.mat'), mdict={'im':im_stack})
    np.savetxt(os.path.join(out_dir,'im_stack.gz'),im_stack,delimiter=',')
