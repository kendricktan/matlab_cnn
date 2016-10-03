% Download the compressed data set from the following location
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
outputFolder = fullfile('caltech101'); % define output folder

% Only download dataset once
if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

% Use 3 categories for now
rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'airplanes', 'ferry', 'laptop'};

% Load and normalize images
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)normalize(filename);
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

% Load our trained neural network
cnnMatFile = fullfile('imagenet-caffe-alex.mat');
convnet = helperImportMatConvNet(cnnMatFile);

% Extract out our feature layers
featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% New image
newImage = fullfile(rootFolder, 'airplanes', 'image_0690.jpg');

% Pre-process the images as required for the CNN
img = readAndPreprocessImage(newImage);

% Extract image features using the CNN
imageFeatures = activations(convnet, img, 'fc7');

% Make a prediction using the classifier
label = predict(classifier, imageFeatures);