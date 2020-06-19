# FoldFinder
This repository contains code for the FoldFinder web app,
which classifies user-uploaded photos of origami and provides instructions so that the user can replicate origami they especially like.

The app is live on AWS and accessible at [foldfinder.xyz](www.foldfinder.xyz).

For a demo and evaluation of the app, see {YT_link TK}.

## Classifiers
The `Classifiers` directory contains the two convolutional neural network models that power the app (stored as `.h5` files), alongside Jupyter Notebooks that walk through the model construction process.
Note that, due to continued experimentation, the outputs of the Notebooks do not precisely match the final models.

The contents of the `ori_not` subdirectory pertain to a "detector" that distinguishes origami from other kinds of images (represented in model training by [the COIL-100 dataset](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)).

The contents of the `origami_classifier` subdirectory pertain to a "classifier" that assigns an image of origami to one of eight classes: butterfly, crane, dragon, duck, frog, lotus, peacock, or star.
In addition to the model and accompanying Notebook, this subdirectory also contains a pickled dictionary that enables conversion from numerical model outputs to text labels.

## Flask
The `Flask` directory contains the files that run the FoldFinder web app.
The required packages are listed in `requirements.txt`.
Most of the inner workings of the app may be found in `/FoldFinder/views.py`, the `/FoldFinder/templates/` subdirectory contains all of the `.html` templates for the app's pages, and the `/FoldFinder/static/` subdirectory contains a variety of materials auxiliary to the `.html` files' functionalities.
