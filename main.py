import tensorflow as tf
import numpy as np
import cv2, os
import matplotlib.pyplot as plt

#--------------utility functions-------------------------
def load_and_process_img(path_to_img, img_size=512):
    img = cv2.imread(path_to_img)
    if img is None:
        raise FileNotFoundError(f"Could not read image from path: {path_to_img}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

#-------------------style transfer model-------------------------------
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [vgg.get_layer(name).output for name in content_layers]
        model_outputs = style_outputs + content_outputs

        self.vgg = tf.keras.Model(vgg.input, model_outputs)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        outputs = self.vgg(inputs)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {name: value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name: value for name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

def total_variation_loss(image):
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

def run_style_transfer(content_img_np, style_img_np, iterations=1500, content_weight=1e4, style_weight=1e-2, img_size=512):

    def _process_image(img_np):
        img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return tf.convert_to_tensor(img / 255.0)

    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    extractor = StyleContentModel(style_layers, content_layers)
    content_image = _process_image(content_img_np)
    style_image = _process_image(style_img_np)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    generated_image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            
            content_loss = tf.add_n([tf.reduce_mean((content_targets[name] - outputs['content'][name])**2) 
                                     for name in content_targets.keys()])
            
            style_loss = tf.add_n([tf.reduce_mean((style_targets[name] - outputs['style'][name])**2) 
                                   for name in style_targets.keys()])
                                   
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss += 30 * total_variation_loss(image)

        grad = tape.gradient(total_loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    for i in range(iterations):
        train_step(generated_image)
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i+1}/{iterations}")

    final_tensor = generated_image[0] 
    final_tensor = tf.clip_by_value(final_tensor * 255, 0, 255)
    final_image = tf.cast(final_tensor, tf.uint8).numpy()
    final_image = cv2.bilateralFilter(final_image, d=9, sigmaColor=75, sigmaSpace=75)
    
    return final_image

#------------------main----------------------------
if __name__ == '__main__':

    CONTENT_PATH = 'path/to/content_image.jpg'
    STYLE_PATH = 'path/to/style_image.jpg'
    
    final_tensor = run_style_transfer(CONTENT_PATH, STYLE_PATH, iterations=1000)
    final_image = tensor_to_image(final_tensor)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(final_image)
    plt.axis('off')
    plt.title("Stylized Image")
    plt.show()

    #save the image
    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    final_image = cv2.bilateralFilter(final_image, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite('stylized_image.png', final_image_bgr)