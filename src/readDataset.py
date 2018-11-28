import os
import tensorflow as tf

def load_uiuc_data(dataset_path):
    images=[]
    labels=[]
    for root, dirs, files in os.walk(dataset_path):  
        for dirname in dirs:
            for filename in os.listdir(os.path.join(root,dirname)):
                if filename.endswith("jpg"):
                    img=tf.read_file(os.path.join(root,dirname,filename))
                    image = tf.image.decode_jpeg(img, channels=3)
                    images.append(image)
                    labels.append(dirname)
    images=tf.convert_to_tensor(images)
    labels=tf.convert_to_tensor(labels)

    return [images,labels]
