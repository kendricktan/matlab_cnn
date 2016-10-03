% Setup MatConvNet
addpath ~/Documents/MATLAB/matconvnet-1.0-beta21/matlab
vl_setupnn;

% Load a model and upgrade it to MatConvNet current version.
net = load('imagenet-caffe-alex.mat');
net = vl_simplenn_tidy(net);

% Read and preprocess image
im = imread('peppers.jpg') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% Run the CNN.
res = vl_simplenn(net, im_) ;

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f', net.meta.classes.description{best}, best, bestScore)) ;
