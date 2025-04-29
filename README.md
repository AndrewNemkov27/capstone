This project delves into the language of the Grappler Baki Japanese manga through the implementation of Optical Character Recognition (OCR), a pre-trained machine learning model. Using raw Japanese manga panels as the foundation for the project's dataset, the OCR is used to extract Japanese text from around 4000 manga images into a computer-readable format. 

The culmination of this work can be found in the attatched blog post in the doc folder. To view the project blog, please input the following link into your browser:

https://andrewnemkov27.github.io/capstone/doc/blog.html

Most of the visualization of the project are also contained in the final blog post. Considering this, I chose to not include the below code in the final results, the reason as to which I explain further:

1. OCR Confidence vs Word Length Scatter Plot (not cleaned):
- This graph is an unmodified, earlier version of my analysis on the relationship between the OCR model's confidence and an extracted word's length. For this graph specifically, i did not apply proper coloring and point jittering, making the whole far less informative and visually confusing. For this reason I chose to not include it in the final results of the project.

2. Top 50 Least Frequent Nouns Combined Bar Graph:
- This bar graph shows the 50 least frequent nouns for the three Baki manga series I used in my dataset. The results of visalizing the most rare words in the three series provides little meaninful information on the focus of the project. Due to this reason, I choose to exclude it from the final blog post.

3. Dataset Creation Code:
- In addition to the two excluded figure above, I will also not include any of my dataset creation code in the final blog post. The code required approximately 6 hours to fully run, making the process time-consuming and impractical for the viewer to execute each time they want to run the project. Also, it is important to note that the OCR data creation process took many iterations to fully complete, encountering many issues relating to computer memory and Visual Studio Code's performance. Given this information, I have kept all my data creation code as python files as seen by their paths below.
- capstone/(1)-main/(1)-codeAndData/dataCreation.py
- capstone/(1)-main/(1)-codeAndData/dataFreq.py

4. Testing Code:
One final thing to note regarding exclusion of code is that the testing code for the OCR model is also not shown in the final blog post. Even though this process is relatively short to run (around 5 minutes total) it is still time-consuming. Since this project's main goal was on analysis rather than testing, it would not make sense to extensively overview it in the blog. I have kept all of my this code of test images, test dataset, and test code in the folders found in the below section of the project below.
- capstone/(0)-test