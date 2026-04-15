# DeepFake_thesis

## FaceForensics++
I have filled an official form in order to get the script for downloading the dataset's videos
I saved the script in the file faceforensics/download.py and run the following command:
original videos: python download.py data -d original -c c23 -t videos --server EU2 
deepfake videos: python download.py data -d Deepfakes -c c23 -t videos --server EU2 

it saves the original videos in c23 compression in the "data" folder
I use c23 compression because it keeps the videos in high quality (h264)

The overall dimension of the dataset compressed with c23 is 10GB (original 1000 videos + manipulated 4000 videos) - for now I am using only the original and Deepfakes videos (so 5GB)

Then for each video I have sampled randomly 10 frames in the jpg format (extract_random_frames.py) 

## Dataset class

I have defined the FFDataset class that returns the couples (frame,label).
In particular it returns the frames from the official split json files for train, validation and test set of the FF++ dataset.

These json files contain the coupled videos indexes [indx_1, indx_2] such that
- the indx_1 and indx_2 videos are the original ones
- the indx_1 video is manipulated using the indx_2 face
- the indx_2 video is manipulated using the indx_1 face

## Metrics
I have created a class that computes at each epoch accuracy, precision, recall, f1 score, ROC-AUC, average l2 and linf norms and saves their history through the epochs for plotting

## Training
- In train_clean.py there is the clean version of the training, so with no adversarial robustness, just for the task of deep fake detection
- In train_robust_FGSM.py performs the adversarial training on FGSM with a final loss function which includes by 50% the clean loss and by 50% the adversarial loss.
- In train_robust_SQUARE.py the training that makes the model robust on the SQUARE black box attack is still an adversarial training on FGSM with the addition of a entropy penalty.
Since Square attack is black box and does not use the gradient, it looks for the regions where small perturbations may change the prediction, so it exploits sharp decision boundaries. The hypotesis is that the adding of entropy penalty makes the probability distribution more smooth. This can improve query-based black box attacks but it does not provide strong robustness against gradient-based attacks.

The robust models has been trained also with two epsilon schdulers, that manage how the FGSM epsilon is incremented during the training, instead of using a fixed one for the overall process. In particular a linear and a cosine scheduler has been used.
So the final training file is train_robust.py, which is a more compact code for both train_robust_FGSM.py and train_robust_SQUARE.py and that handles both training with and without epsilon scheduler.

## Testing
In test_fgsm.py and test_square.py the models are tested on clean, FGSM and Square images and then the metric of all the three cases are compared in order to understand the level of generalizability. In particular the accuracy, attack success rate and AUC score are considered.
The number of images that have been used for these tests are:
- 1000 for FGSM attack
- 1000 for SQUARE attack
For both attacks the results are compared to those obtained testing on the clean images with the corresponding number of images.

In test_attacks.py there is a function test_attack that generalizes the testing process for the other attacks, in particular FGSM, IFGSM, PGD, JSMA, Gen Attack, NES and ZOO.

In attacks.py the attacks functions implemented from art and foolbox libraries are initialized such that they will be called in the test_attack function.

run_test.py just runs the tests for the chosen models.




