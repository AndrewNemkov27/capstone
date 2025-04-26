This project delves into the language and word usage found in the Grappler Baki manga through the usage of Optical Character Recognition (OCR), a pre-trained machine learning model. Using raw Japanese manga panels as the foundation for the project's dataset, the OCR was used to extract Japanese text from around 4000 manga images with each phrase including its OCR confidence score. In addition, this result was cleaned formatted to allow for proper analysis of frequency count, word length statistics, parts of speech distribution, and more. 

The culmination of this work can be found in the attatched blog post in the doc folder.

Most of the visualization of the project are also contained in the final blog post. They help to lay a path of my data analysis, showing my process of working with my dataset. Considering this, I chose to not include two figures which i explain further below:

1. OCR Confidence vs Word Length Scatter Plot (not cleaned):
- This graph is an unmodified, earlier version of my analysis on the relationship between the OCR model's confidence and an extracted word's length. For this graph specifically, i did not apply proper coloring and point jittering, making the whole far less informative and visually confusing. For this reason I chose to not include it in the final results of the project.

2. Top 50 Least Frequent Nouns Combined Bar Graph:
- This bar graph shows the 50 least frequent nouns for the three Baki manga series I used in my dataset. The results of visalizing the most rare words in the three series, as I guessed, provides little to no meaninful information on the Baki story as a whole. Due to this reason, and the graphs generally unappealing look I choose to exclude it from the final blog post.