% Setup MatConvNet
addpath ~/Documents/MATLAB/matconvnet-1.0-beta21/matlab
vl_setupnn;

% Load a model and upgrade it to MatConvNet current version.
disp('Loading model...');
net = load('imagenet-caffe-alex.mat');
net = vl_simplenn_tidy(net);
disp('Finished loading model...');

% Open Webcam and preview it
cam = webcam();
preview(cam);

% Audio recorder
recordObj = audiorecorder;

% Wait for clapping sound
while 1    
    disp('Waiting for clapping sound...');
    while 1
        recordblocking(recordObj, 1);
        y = getaudiodata(recordObj);
        if max(y) > 0.75
            break
        end
    end
    disp('Clapping sound heard, capturing image...');

    % Read and preprocess image
    im = snapshot(cam);
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;

    % Run the CNN.
    res = vl_simplenn(net, im_) ;

    % Show the classification result.
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    figure(1) ; clf ; imagesc(im) ;
    title(sprintf('%s, score %.3f', net.meta.classes.description{best}, bestScore));
    
    %do_again = input('Record image again? (y/n): ', 's');
    %if do_again == 'n'
    %    break
    %end
end

closePreview(cam);