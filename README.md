## Python Programming for Data Science II <img src="oudce_logo.png" align="right"/>

### Massimiliano Izzo 

Materials for [Python Programming for Data Science WOW Course](https://www.conted.ox.ac.uk/courses/python-programming-for-data-science-part-1?code=O21P656COW) - **this page will be updated as the course progresses**.

The class workspace on **Slack** is https://pp4ds-ox.slack.com. I encourage you to ask questions should you have them in the Slack channel incase your classmates can help. Massi (your tutor; massimiliano.izzo@conted.ox.ac.uk) will also check Slack and provide support where possible. Download Slack from: https://slack.com/get

If you do not wish to use slack you can use Canvas to contact me and other students. 

To use **Jupyter** yourself, I recommend you download and install **Anaconda**, a Python Data Science Platform, from: [here](https://www.anaconda.com/products/individual) Make sure you download the **Python 3** version of Anaconda, ideally Python 3.8 or more recent. You can also install Jupyter if you have a standard Python distribution installed. Ask your tutors for assistance if you need to install Jupyter on your own machine.

To get the contents of this repository I recommend that you install **Git SCM**, a source code management software, that will help you keep up-to-date with the repository. I will be adding content as the course progresses and Git will allow you to pull new material as it becomes available.

### Cloning this repository to use it offline

If you want to run the notebooks on your own computer at home, apart from installing Jupyter/Anaconda as per above, you will need to install **Git**, which is a source code management software, from [here](https://git-scm.com/downloads). Windows users can also get Git here: https://gitforwindows.org/. Once installed, you need to open up the command-line ("Command Prompt" on Windows or "Terminal" on Mac OSX) to run some commands.

Change directory to somewhere sensible, such as `My Documents` or similar on Windows or `Documents` on Mac OSX. Assuming you're using `Documents`:

```
cd Documents
```

Then ask Git to clone this repository with the following command.
```
git clone https://gitlab.com/data-science-course/pp4ds-pt2-tt2022.git
```
or, if you have SSH enabled
```
git clone git@gitlab.com:data-science-course/pp4ds-pt2-tt2022.git
```

This will create a subdirectory called `pp4ds-pt2-tt2022` in your `Documents` folder. When you need to update the content at some later time after I have added some new files to the repository, you will need to open up the command-line again and do the following commands.
```
cd Documents/pp4ds-pt2-tt2022
git pull
```
What this does is to ask Git to check if there are any new changes in the online repository and to download those new files or updates to the existing files.

Either some lines of stuff should whizz by, or it will say `Already up to date.` if there are no new changes.

If this doesn't work, you may need to force the update, which will overwrite your local files. To do this (make sure any of your own work is renamed or moved outside of the `pp4ds-pt2-tt2022` folder first):
```
git fetch --all
git reset --hard origin/master
```

### Course Programme (inidicative)

**Week 1:** Introduction to the course. Basic overview of Machine Learning. Linear Regression example.

**Week 2:** Overview of a data-science preprocessing pipeline.

**Week 3:** Supervised Learning: regression.

**Week 4:** Supervised Learning: classification.

**Week 5:** Decision Trees. Ensemble Methods. The Perceptron.

**Week 6:** Deep Learning: Feed-forward Neural Networks.

**Week 7:** Deep Learning: Convolutional Neural Networks (CNNs) for Image Processing. Recurrent Neural Networks (RNNs) for time series analysis.

**Week 8:** Dimensionality Reduction and Unsupervised Learning.

**Week 9:** Natural Language Processing (NLP): an overview. Word embeddings. RNNs for NLP.  Attention-based models  (Transformers).

**Week 10:** Autoencoders and Generative Adversarial Networks (GANs). Explainability of ML models.

## Week 1: Introduction to the course

* Lecture notes (face to face course): [download](https://tinyurl.com/2k3rjjna) 
* Intro Exercise 01A: **Notebook Basic** 
* Intro Exercise 01B: **Running Code** 
* Intro Exercise 01C: **Working with Markdown** 
* Intro Exercise 01D: **Notebook Exercises** 
* Exercise 01: **Matrix Operations recap, Linear Regression** (solutions can be found in the solutions folder)

## Week 2: Data Science Pipeline for machine learning

* Lecture notes (face to face course): [download](https://tinyurl.com/2p9xzj57)
* Live Demo: The Kings County Dataset (see `live-demos` and `datasets` directory)

## Week 3: Regression

* Lecture notes (face to face course): [download](https://tinyurl.com/6e3c993n)
* Live Demo: The Kings County Dataset (same as the week before)
* First assignment (face-to-face course): see the `assignments` directory.

## Week 4: Intro to Classification

* Lecture notes (face to face course): [download](https://tinyurl.com/4nk5kmst)
* Live Demo: The MNIST Dataset (see `live-demos-*` and `datasets` directory)

## Week 5: More Classification

* Live Demo: The MNIST Dataset

## Week 6: Decision Trees. Ensemble Methods. 

* Lecture notes (face to face course): [download](https://tinyurl.com/ye2x3xuz)
* Live Demo: Decision Trees with the IRIS dataset (see `live-demos-*` and `datasets` directory)
* Live Demo: Ensemble Methods (see `live-demos-*` and `datasets` directory)

## Week 7. Hyperparameter Tuning. Dimensionality Reduction. 

* Lecture notes (face to face course): [download](https://tinyurl.com/2hxjp7sr)
* Live Demo: Dimensionality Reduction. PCA. UMAP. (see `live-demos-*` and `datasets` directory)


## Week 8: Unsupervised Learning. Intro to Neural Networks.

* Lecture notes (face to face course): [download](https://tinyurl.com/2dx6ac5j)
* Live Demo: Unsupervised Learning. Clustering (see `live-demos-*` and `datasets` directory)


## Week 9: Fully-connected Neural Networks. Deep Learning Intro

* Lecture notes (face to face course): [download](https://tinyurl.com/f5t4kmyr)
* Live Demo: Intro to Neural Networks (see `live-demos-*` and `datasets` directory)
* Live Demo: Training Deep Learning Models (see `live-demos-*` and `datasets` directory)


## Week 10: CNNs, RNNs and final remarks

* Lecture notes (face to face course): [download](https://tinyurl.com/uj97cv77)
* Live Demo: Convolutional Neural Networks (see `live-demos-*` and `datasets` directory)
* Live Demo: Recurrent Neural Networks (see `live-demos-*` and `datasets` directory)
