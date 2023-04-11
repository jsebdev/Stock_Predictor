from dotenv import load_dotenv
load_dotenv('.env')

import gradio as gr
from utils import *
import os


days_to_plot = 50

data = get_data().iloc[-500:]


data_to_plot = data.iloc[-days_to_plot:][["Close"]]
data_to_plot['date'] = data_to_plot.index

with gr.Blocks() as demo:
    gr.Markdown("# Apple Predictor")
    predict_button = gr.Button("Predict")
    with gr.Row() as row0:
        with gr.Column() as col0:
            gr.Markdown("## Last candle info")
            last_open = gr.Textbox(get_last_candle_value(data, 'Open') ,label="Last Open")
            last_max = gr.Textbox( get_last_candle_value(data, 'High') ,label="Last Max")
            last_min = gr.Textbox( get_last_candle_value(data, 'Low') ,label="Last Min")
            last_close = gr.Textbox( get_last_candle_value(data, 'Close') ,label="Last Close")

        with gr.Column() as col1:
            gr.Markdown("## Next Candle Prediction")
            jump_text = gr.Textbox(label="Jump")
            open_text = gr.Textbox(label="Open")
            max_text = gr.Textbox(label="Max")
            min_text = gr.Textbox(label="Min")
            next_close_text = gr.Textbox(label="Close")
    with gr.Row() as row1:
        value_plot = gr.LinePlot(data_to_plot,
                                 x="date",
                                 y="Close",
                                 label=f'Last {days_to_plot} days',
                                 y_lim=[float(data_to_plot['Close'].min())-5, float(data_to_plot['Close'].max())+5])

    outputs = [jump_text,
               open_text,
               max_text,
               min_text,
               next_close_text
               ]
    predict_button.click(lambda: predict(data), outputs=outputs)

demo.launch(debug=True)
