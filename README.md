# Llamalama_II

## Setup Environment

### Install Dependencies
Make sure you have Python installed. Then, install the required dependencies by running:

```sh
pip install streamlit torch langchain-community chromadb pydantic python-dotenv rvc-python ollama streamlit-mic-recorder gTTS unstructured openai-whisper
```

## Running the Application

### Without ngrok
To run the application locally without ngrok, use the following command:

```sh
streamlit run stream.py --server.port 80
```

### With ngrok do not do this contract kun.kerdthaisong@gmail.com first
If you need to expose the application to the internet using ngrok, follow these steps:

1. Start an ngrok tunnel:
   ```sh
   ngrok http 80
   ```
   This will generate a public URL (e.g., `https://feline-steady-quickly.ngrok-free.app`).

2. Run the Streamlit app:
   ```sh
   streamlit run stream.py --server.port 80
   ```


