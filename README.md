# ChatBot With Stream
This is a chatbot with real time stream like ChatGpt.

## Setup
  
  clone project with Python 3.10.10
  ```code
  git clone https://github.com/USTAADCOM/chatbot_stream.git
  cd chatbot_stream
  pip install -r requirements.txt
  ```
## Project Structure

```bash
chatbot_stream
   │   app.py
   │   requirements.t
   │
   ├───static
   │       chat.js
   │       style.css
   │
   └───templates
           chat.html
```

## Run FlaskServer  
```code
python app.py
```
Visit URL 
```code
http://127.0.0.1:5000/chat
```
## Run Gradio Server   
```code
python gradio_server.py
```
Visit URL 
```code
http://127.0.0.1:7860
```
## Note
Change the Port number above with your port number.
## Gradio Demo On Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USTAADCOM/chatbot_stream/blob/main/ChatBot_Streaming.ipynb)

