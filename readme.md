# Fake Review Checker
[Live Site](https://fake-review-checker-experimental.streamlit.app/)

## Introduction

This web application classifies if the review(s) provided via text or file in CSV or JSON format are real or fake. A confidence rating is provided upon classification, determining how confident the mode is in classifying the reviews.

## Features

### Text Input Review

![Text Input Feature](/images/text_input.png)

Users are able to manually input text in the text area. Upon clicking the button, the review is classified by the model and a confidence rating is given.

### File Upload Review

![File Upload Feature](/images/file_upload.png)

Users are able to upload a CSV or JSON file to bulk check reviews. A dropdown option to select the review column will be provided if the reviews column isn't found. Once the reviews have been classified, a table will show up with each row containing the review, the classification, and the confidence rating. There is also an option to download the results in a CSV file.

## Credits
[Fake Review Dataset by Mexwell](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset/data)
