# Readme

This directory contains all materials related to Weiß 2015: "On the Applicability of Features of Linguistic Complexity to the Classification of Political Speeches", which was written as final project for the main seminar "Machine Learning for Computational Linguists".

The paper evaluates the applicability of linguistic complexity features to the tasks of party and government affiliation classification in political speeches from German *Bundestag* by comparing them to word embeddings. Using a linear SVM word embeddings outperform classification performance of complexity features, which, however, perform above chance.
Different complexity dimensions exhibit intriguing performance differences and give evidence, that especially morphological and lexical complexity structurally varies across parties and government vs. non-government speeches. 

The directory is structures as follows:

* **zweiss2016-ml4cl.pdf** is the paper Weiß 2015: "On the Applicability of Features of Linguistic Complexity to the Classification of Political Speeches".
* **bundesparser-complexity.csv.gz** contains the complexity table used for the experiment.
* **./data-preprocessing/** contains the Python scripts used to extract plain text speeches from the PolMine corpus.
* **./ml-analysis/** contains the Python scripts used for the machine learning analysis.


