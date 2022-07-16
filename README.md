# machine-learning-model-to-assign-a-gender-to-first-names

## Setting up your computer if you are new to running Python code

1) Download the zip file (from this Github page) to your computer and unzip it.

2) Install an appropriate version of Anaconda from [https://www.anaconda.com/products/distribution#Downloads](https://www.anaconda.com/products/distribution#Downloads).

3) Start Anaconda Navigator on your computer.

4) Within Anaconda Navigator, look for Spyder and launch it.

5) Within Spyder, on the menu bar at the top, select "Tools" and then "Preferences".

6) On the left panel, look for "Current working directory" and select it.

7) On the right panel, under "Startup", select the radio button for "The following directory".

8) Click the icon on the right and select the directory (also called folder) where you saved the unzipped files in Step 1 above.

## Requirements

To install requirements:

9) Start Anaconda Prompt on your computer.

10) Type the following command and hit "Enter":

```
pip install -r requirements.txt
```

## Training

To train the model, run the following command (in Anaconda Prompt) and hit "Enter":

```
python train.py
```

You would see the accuracy reported for the test set (which is a subset of the data in "name_gender.csv").

## Evaluation

To determine the gender of names which have not been used to train the model, run the following command (in Anaconda Prompt) and hit "Enter":

```
python evaluate.py
```

You would see the accuracy reported for the names in "unseen_names.csv".
