FaceXpert Guard is a facial recognition system I built from scratch using Siamese neural networks. I coded the bulk of the machine learning in Jupyter Notebook with basic opencv to capture images of the users face. The data is then preprocessed,and then augmented (random brightness changes, contrast adjustments, flipping). The images are processed through an embedding layer turning them into 3d vectors, then put through an L1 distance function to measure the Manhattan distance. The model is saved into your project folder. Once training was done, a VS Code app was created using tkinter where the final user interface was built.

Usage:

**note to user:**the project was initially coded using the lfw(labeled faces in the wild) dataset for negative images. However, the website has recently gone down at the time of the creation of this repo. A seperate set of images must be manually added into the negative data folder before usage. I will update the repo if the website comes back online and its data is available. 

Install python3 and jupyter lab
Download the repository and run the command "juypter lab" in command prompt.
Open the the ipynb file and run (this can take significant time if your computer does not have a powerful GPU)
Open app.py under the app folder and add in the file paths of your folder.
Run the app and test verification

Common Errors:
If the ipynb file given does not work when launched in juptyer lab, convert the .py script given using the jupyter conversion script and use the converted script in jupyter lab instead.

