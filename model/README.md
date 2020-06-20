To train model following steps should be performed:
* download available parallel corpus
* build dictionary (ru and zh)
* (back translation) train ru->zh model
* (back translation) infer ru-zh model on monolingual corpus
* convert corpus to TFRecords
* upload TFRecords to appropriate storage
* train model
