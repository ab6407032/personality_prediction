# Personality Prediction Model

This repository contains the implementation of a machine learning model for predicting personality traits based on textual data.

## Files in the Repository

- `app.py`: Main application script.
- `model.py`: Contains the model architecture and training script.
- `label_encoder.pkl`: Pickled Label Encoder used for encoding target labels.
- `personality_model.h5`: Trained model file.
- `personality_model_lstm.h5`: Trained LSTM model file.
- `requirements.txt`: List of Python packages required for the project.
- `tfidf_vectorizer.pkl`: Pickled TF-IDF Vectorizer used for transforming text data.
- `tokenizer.pkl`: Pickled Tokenizer used for text preprocessing.
- `UpdatedResumeDataSet.csv`: Dataset used for training the model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/personality-prediction.git
    cd personality-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Make sure you have all the necessary files in the repository.
2. Run the application:
    ```bash
    python app.py
    ```

## Dependencies

The `requirements.txt` file lists all the dependencies needed to run this project. Here are the main packages used:

- absl-py==2.1.0
- astunparse==1.6.3
- blinker==1.8.2
- certifi==2024.6.2
- charset-normalizer==3.3.2
- click==8.1.7
- colorama==0.4.6
- Flask==3.0.3
- flatbuffers==24.3.25
- gast==0.6.0
- google-pasta==0.2.0
- grpcio==1.64.1
- h5py==3.11.0
- idna==3.7
- itsdangerous==2.2.0
- Jinja2==3.1.4
- joblib==1.4.2
- keras==3.4.1
- libclang==18.1.1
- Markdown==3.6
- markdown-it-py==3.0.0
- MarkupSafe==2.1.5
- mdurl==0.1.2
- ml-dtypes==0.3.2
- namex==0.0.8
- nltk==3.8.1
- numpy==1.26.4
- opencv-python==4.10.0.84
- opt-einsum==3.3.0
- optree==0.11.0
- packaging==24.1
- pandas==2.2.2
- protobuf==4.25.3
- Pygments==2.18.0
- python-dateutil==2.9.0.post0
- pytz==2024.1
- regex==2024.5.15
- requests==2.32.3
- rich==13.7.1
- scikit-learn==1.5.0
- scipy==1.14.0
- six==1.16.0
- tensorboard==2.16.2
- tensorboard-data-server==0.7.2
- tensorflow==2.16.2
- tensorflow-intel==2.16.2
- tensorflow-io-gcs-filesystem==0.31.0
- termcolor==2.4.0
- threadpoolctl==3.5.0
- tqdm==4.66.4
- typing_extensions==4.12.2
- tzdata==2024.1
- urllib3==2.2.2
- Werkzeug==3.0.3
- wrapt==1.16.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
