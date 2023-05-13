from transformers import pipeline
import gradio as gr

model = pipeline("summarization")

def predict(text):
    return model(text)[0]['summary_text']


with gr.Blocks() as block:
    textbox = gr.Textbox(placeholder="Paste your text here...", lines=5, label="Input Text")
    gr.Interface(fn=predict, inputs=textbox, outputs="text", title="Summarizer", description="Summarize your text with Hugging Face Transformers", article="https://huggingface.co/transformers/model_doc/bart.html")

block.launch()