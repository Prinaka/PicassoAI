# PicassoAI

This is a deep learning application that combines the content of one image (such as a photograph) with the creative style of another (like a painting) using a pretrained CNN. The CNN acts as a feature extractor to separately represent the content and style of both images. The result is a brand-new image that retains the original scene's structure while being "painted" in the aesthetic of the second image.

Live Demo : https://huggingface.co/spaces/Prinaka/Image-Style-Transfer

**Key Features:**
* Image Synthesis: Combine the content of any photograph with the artistic style of a famous painting or a custom image.
* Highly Customizable: Easily adjust parameters such as the number of iterations, content weight, and style weight to achieve a desired effect.
* PyTorch/TensorFlow-Based: Built on a robust and widely-used deep learning framework.

**Installation**

1. Clone the repository:

```
git clone https://github.com/Prinaka/PicassoAI.git
cd PicassoAI
```
2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```

**Usage**

To run the style transfer, use the following command structure. Make sure your input images are in the project directory or provide their full path.
```
python main.py --content_image <content_image_path.jpg> --style_image <style_image_path.jpg> --output_image <output_image_path.jpg>
```

**License:**

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


**Credits:**

Based on the paper '[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)' by Gatys et al.

