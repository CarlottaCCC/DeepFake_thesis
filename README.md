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