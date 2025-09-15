import gradio as gr
from main import StyleContentModel, run_style_transfer

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
extractor = StyleContentModel(style_layers, content_layers)

#--------gradio app------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Neural Style Transfer")
    gr.Markdown("Combine the content of one image with the artistic style of another. This demo uses a VGG19 model. Processing can take a minute, especially on CPU.")

    with gr.Row():
        content_img = gr.Image(label="Content Image", type="numpy")
        style_img = gr.Image(label="Style Image", type="numpy")
    
    run_button = gr.Button("Generate Image", variant="primary")
    output_img = gr.Image(label="Result")
    
    run_button.click(
        fn=run_style_transfer, 
        inputs=[content_img, style_img], 
        outputs=output_img
    )
    
    gr.Markdown("---")
    gr.Markdown("Based on the paper '[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)' by Gatys et al.")

#Launch the Gradio app
demo.launch(share=True,debug=True)