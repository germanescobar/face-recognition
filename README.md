# Face recognition

A toy project I'm using to learn about face recognition using Python and the OpenCV library. It is composed of four scripts:

* `capture.py` - Captures an image from your camera, crops the faces and saves them as 100x100 grayscale images.
* `photo.py` - Loads images from a folder, crops the faces and saves them as 100x100 grayscale images in the `data/negative/` folder.
* `train.py` - Trains the algorithm to recognize an image. The output is an XML file called `training-data.xml`
* `predict.py` - Captures an image from your camera, crops the face and runs the prediction algorithm.

## Usage

**Note:** This only works in OSX.

1. Take some pictures of yourself using the `capture.py` script and save them to the `data/positive/` folder. Take 5 pictures: normal, smiling, sad, frowning, and surprised.
2. Download some pictures of your friends, save them to the `photos/` folder and run the `photo.py` script (10 to 15 faces are enough).
3. Execute the `train.py` script to generate the XML.
4. Execute the `predict.py` and smile! If it recognizes you, it will say hello.


## TODO

* Fix `capture.py` to create new files when there are multiple faces on the captured image and save them directly to the `data/positive/` folder.
* Create a decent CLI.