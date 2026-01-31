# End-to-End Chatbot using Python and Streamlit

This is a simple end-to-end chatbot application built using Python, scikit-learn, and Streamlit.

## Description

The chatbot is a simple rule-based chatbot that uses a machine learning model to understand the user's intent and provide a relevant response. The model is trained on a small dataset of intents and patterns. The user interface is built with Streamlit, which allows for a simple and interactive web application.

## Technologies Used

*   **Python**: The core programming language for the application.
*   **scikit-learn**: Used for building the machine learning model (TF-IDF Vectorizer and Logistic Regression).
*   **Streamlit**: Used for creating the web user interface.
*   **NLTK**: Used for natural language processing tasks (tokenization).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the NLTK `punkt` tokenizer:**
    The first time you run the application, it will download the `punkt` tokenizer from NLTK. This is a one-time process.

## How to Run

To run the chatbot application, use the following command:

```bash
streamlit run filename.py
```

This will start the Streamlit development server and open the application in your web browser.

## Deployment

This application can be easily deployed using [Streamlit Community Cloud](https://streamlit.io/cloud).

### Steps to Deploy on Streamlit Community Cloud:

1.  **Push your code to a GitHub repository.**
2.  **Sign up for a Streamlit Community Cloud account.**
3.  **Click on the "New app" button on your Streamlit Cloud dashboard.**
4.  **Connect your GitHub account and select the repository you want to deploy.**
5.  **Make sure the main file path is set to `app.py`.**
6.  **Click on the "Deploy!" button.**

Your chatbot application will be deployed and accessible via a public URL.
