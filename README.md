# Fake News Classification using LSTM and Bidirectional LSTM RNN

This project focuses on classifying news articles as real or fake using Long Short-Term Memory (LSTM) and Bidirectional LSTM Recurrent Neural Networks (RNNs). The models are trained on a dataset of labeled news articles to distinguish between legitimate news sources and fake news sources.

## Dataset

The dataset consists of a collection of news articles labeled as real or fake. Each article is preprocessed and tokenized before being fed into the LSTM and Bidirectional LSTM RNN models.

## Preprocessing

Text preprocessing techniques such as tokenization, removing stopwords, and stemming are applied to clean the text data before training the models. Additionally, word embeddings may be utilized to represent words in a dense vector space.

## Model Architectures

### LSTM (Long Short-Term Memory)

LSTM networks are a type of recurrent neural network capable of learning long-term dependencies in sequential data. In this project, LSTM networks are employed to analyze the sequential nature of news articles and make predictions about their authenticity.

### Bidirectional LSTM RNN

Bidirectional LSTM RNNs enhance the capabilities of traditional LSTM networks by processing input sequences in both forward and backward directions. This allows the model to capture contextual information from both past and future states, leading to improved performance in text classification tasks.

## Results

The performance of the LSTM and Bidirectional LSTM RNN models is evaluated using metrics such as accuracy. Comparative analysis is conducted to assess the effectiveness of each model in distinguishing between real and fake news articles.

## Dependencies

The project is implemented in Python using the following libraries:

- TensorFlow
- Keras
- NumPy
- NLTK (Natural Language Toolkit)
- scikit-learn
- Matplotlib

## Kaggle Link
https://www.kaggle.com/code/dsingh9/lstm-bidrectional-lstm-fake-news-classifier/notebook

## License

This project is licensed under the [MIT License](LICENSE).
