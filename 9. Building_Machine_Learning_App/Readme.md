# Sentiment Analysis Web App

This project is a simple web application that analyzes the sentiment of a given text. It uses a pre-trained DistilBERT model from Hugging Face to classify the input text as either POSITIVE or NEGATIVE. The user interface is built with Gradio, providing a straightforward way to interact with the model.

## Technologies Used

*   Python
*   Gradio
*   Hugging Face Transformers
*   PyTorch

## Setup and Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Install the required libraries:**
    Create a virtual environment (recommended) and install the necessary packages.
    ```bash
    pip install gradio transformers torch
    ```

## How to Run the Project

1.  **Run the application:**
    Execute the `app.py` script from your terminal.
    ```bash
    python app.py
    ```

2.  **Access the application:**
    The terminal will display a local URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser to use the sentiment analysis tool.

3.  **Use the app:**
    Enter a sentence into the textbox and the model will return its predicted sentiment (POSITIVE or NEGATIVE) along with a confidence score.

## Project Structure

```
.
├── app.py                    # The main Gradio application script.
├── ML_app.ipynb              # Jupyter Notebook with the same application logic.
├── Readme.md                 # This file.
├── .gradio/                  # Directory for Gradio-specific data (e.g., flagged samples).
└── Sentimental_Analysis_Project/
    ├── sentiment_model.pkl   # A trained sentiment analysis model.
    ├── tokenizer.pkl         # A corresponding tokenizer for the model.
    ├── sentimental.ipynb     # Jupyter Notebook for training the custom model.
    └── train data/             # Training and testing data for the custom model.
```
