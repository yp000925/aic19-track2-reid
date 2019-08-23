%******************************************************************************************************************%
% The AIC19 benchmark is captured by 40 cameras in real-world traffic surveillance environment.                    %
% A total of 666 vehicles are annotated. 333 vehicles are used for training. The remaining 333 vehicles are for    %
% testing.                                                                                                         %
% There are 56277 images in total. 18290 images are in the test set, and 36935 images are in the training set.     %
%******************************************************************************************************************%

Content in the directory:
1. "image_query/". This dir contains 1052 images as queries. 
2. "image_test/". This dir contains 18290 images for testing. 
3. "image_train/". This dir contains 36935 images for training. 
4. "name_query.txt". It lists all query file names.
5. "name_test.txt". It lists all test file names.
6. "name_train.txt". It lists all train file names.
7. "test_track.txt" & "test_track_id.txt". They record all testing tracks. Each track contains multiple images of the same vehicle captured by one camera.
8. "train_track.txt" & "train_track_id.txt". They record all training tracks. Each track contains multiple images of the same vehicle captured by one camera.
9. "train_label.xml". It lists the labels of vehicle ID and camera ID for training.
10. "train_label.csv". It lists the labels of vehicle ID in CSV format for training. 
11. "tool/visualize.py" & "tool/dist_example/". It is a Python tool for visualizing the results of vehicle re-identificaition, with an example of input data provided. 
12. "DataLicenseAgreement_AICityChallenge.pdf". The license agreement for the usage of this dataset. 

If you have any question, please contact aicitychallenge2019@gmail.com.
