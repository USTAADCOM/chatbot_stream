import os

import gradio as gr

from text_generation import Client, InferenceAPIClient

def get_client(model: str):
    return InferenceAPIClient(model, token=os.getenv("HF_TOKEN", None))

def get_usernames(model: str):
    """
    Returns:
        (str, str, str, str): pre-prompt, username, bot name, separator
    """
    if model in ("OpenAssistant/oasst-sft-1-pythia-12b", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"):
        return "", "<|prompter|>", "<|assistant|>", "<|endoftext|>"
def predict(
    model: str,
    inputs: str,
    typical_p: float,
    watermark: bool,
    chatbot,
    history):
    client = get_client(model)
    preprompt, user_name, assistant_name, sep = get_usernames(model)
    history.append(inputs)
    past = []
    for data in chatbot:
        user_data, model_data = data

        if not user_data.startswith(user_name):
            user_data = user_name + user_data
        if not model_data.startswith(sep + assistant_name):
            model_data = sep + assistant_name + model_data
        past.append(user_data + model_data.rstrip() + sep)
    if not inputs.startswith(user_name):
        inputs = user_name + inputs
    total_inputs = preprompt + "".join(past) + inputs + sep + assistant_name.rstrip()
    partial_words = ""
    if model in ("OpenAssistant/oasst-sft-1-pythia-12b", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"):
        iterator = client.generate_stream(
            total_inputs,
            typical_p=typical_p,
            truncate=1000,
            watermark=watermark,
            max_new_tokens=500,
        )
    for i, response in enumerate(iterator):
        if response.token.special:
            continue
        partial_words = partial_words + response.token.text
        if partial_words.endswith(user_name.rstrip()):
            partial_words = partial_words.rstrip(user_name.rstrip())
        if partial_words.endswith(assistant_name.rstrip()):
            partial_words = partial_words.rstrip(assistant_name.rstrip())
        if i == 0:
            history.append(" " + partial_words)
        elif response.token.text not in user_name:
            history[-1] = partial_words
        chat = [
            (history[i].strip(), history[i + 1].strip())
            for i in range(0, len(history) - 1, 2)
        ]
        yield chat, history

def reset_textbox():
    return gr.update(value="")


def radio_on_change(
    value: str,
    disclaimer,
    typical_p,
    watermark,
):
    if value in ("OpenAssistant/oasst-sft-1-pythia-12b", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"):
        typical_p = typical_p.update(value=0.2, visible=True)
        watermark = watermark.update(False)
    return (
        disclaimer,
        typical_p,
        watermark,
    )
with gr.Blocks(
    css="""#col_container {margin-left: auto; margin-right: auto;}
                #chatbot {height: 520px; overflow: auto;}""") as demo:
    with gr.Column(elem_id="col_container"):
        model = gr.Radio(
            value="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            choices=[
                "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
                "OpenAssistant/oasst-sft-1-pythia-12b"],
            label = "Model",
            interactive=True)
        chatbot = gr.Chatbot(elem_id="chatbot")
        inputs = gr.Textbox(
            placeholder = "Hi there!", label = "Type an input and press Enter")
        state = gr.State([])
        b1 = gr.Button()
        with gr.Accordion("Parameters", open = False):
            typical_p = gr.Slider(
                minimum = -0,
                maximum = 1.0,
                value = 0.2,
                step = 0.05,
                interactive = True,
                label="Typical P mass")
            watermark = gr.Checkbox(value=False, label="Text watermarking")

    model.change(
        lambda value: radio_on_change(
            value,
            typical_p,
            watermark,
        ),
        inputs=model,
        outputs=[
            typical_p,
            watermark,
        ],
    )

    inputs.submit(
        predict,
        [
            model,
            inputs,
            typical_p,
            watermark,
            chatbot,
            state,
        ],
        [chatbot, state],
    )
    b1.click(
        predict,
        [
            model,
            inputs,
            typical_p,
            watermark,
            chatbot,
            state,
        ],
        [chatbot, state],
    )
    b1.click(reset_textbox, [], [inputs])
    inputs.submit(reset_textbox, [], [inputs])
    demo.queue(concurrency_count=16).launch(debug=True)