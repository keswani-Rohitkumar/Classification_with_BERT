# Sentence Classification with BERT
In this project I turn a pre-trained BERT model into a trainable Keras layer and apply it to the semeval 2017 task 4 subtask B. BERT (Bidirectional Embedding Representations from Transformers) is a new model for pre-training language representations that obtains state-of-the-art results on many NLP tasks. I here demonstrated how to integrate BERT as a custom Keras layer to simplify model prototyping using huggingface.

In this project I have used DistilBERT instead of BERT: DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, and runs 60% faster, while preserving over 95% of BERT’s performance as measured on the GLUE language understanding benchmark.

It is easy to switch between DistilBERT and BERT using the huggingface transformer package. This huggingface package provides many pre-trained and pre-built models that are easy to use via a few lines of code.
Before using DistilBERT or BERT, we need a tokenizer. Generally speaking, every BERT related model has its own tokenizer, trained for that model. Here I have used  the DistilBERT tokenizer from DistilBertTokenizer.from_pretrained function.

Here we are using Four models for text classification.
This first model is the standard way BERT models are used for text classification: we use the embedding of the special [CLS] token at the beginning of the sequence, and put it through a simple classifier layer to make the decision. The TFDistilBertForSequenceClassification class takes care of that for us.
For model1 one we are getting 87% accuracy.

The second model I used was Neural bag of words using BERT.
In this model, we take the NBOW classifier and integrate BERT. Instead of averaging over word2vec or GloVe word vectors, we are averaging over the embedding representations produced by BERT - but otherwise, the classifier is the same. 
The accuracy slighlty decreased to 85.7% for this NBOW model with BERT.

The Third model I tried was adding an lstm layer with BERT, but again the accuracy decreased to 84.5%.

The fourth model I tried was adding a CNN layer with BERT and the accuracy improved to 86.6% but the first standard BERT model achieved the highest accuracy comparing to other three models.

The data can be downloaded using the Sentence_classification_with_BERT.ipynb file.

I have used Google Colab in this project and I have used TPU form Colab (Note: Running the BERT on the CPU would be very slow).
