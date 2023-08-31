from flask import Flask, stream_template, request, Response, render_template
import os
from text_generation import Client, InferenceAPIClient

app = Flask(__name__)
def get_client(model: str):
    """
    string
    """
    return InferenceAPIClient(model, token = os.getenv("HF_TOKEN", None))

def get_usernames(model: str):
    """
    Returns:
        (str, str, str, str): pre-prompt, username, bot name, separator
    """
    if model in ("OpenAssistant/oasst-sft-1-pythia-12b",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"):
        return "", "<|prompter|>", "<|assistant|>", "<|endoftext|>"
def send_messages(messages):
    """
    string
    """
    model = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
    inputs =  messages
    typical_p = 0.2
    client = get_client(model)
    preprompt, user_name, assistant_name, sep = get_usernames(model)
    if not inputs.startswith(user_name):
        inputs = user_name + inputs
    total_inputs = preprompt + "" + inputs + sep + assistant_name.rstrip()
    if model in ("OpenAssistant/oasst-sft-1-pythia-12b", 
                 "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"):
        iterator = client.generate_stream(
        total_inputs,
        typical_p = typical_p,
        truncate = 1000,
        max_new_tokens = 100)
        return iterator

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        text = ""
        messages = request.json['messages']
        def event_stream():
            for response in send_messages(messages = messages):
                if not response.token.special:
                    text = response.token.text
                    yield text
        return Response(event_stream(), mimetype='text/event-stream')
    else:
        return stream_template('./chat.html')

if __name__ == '__main__':
    app.run()
