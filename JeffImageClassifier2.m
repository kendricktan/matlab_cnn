function ImageClassifier2(cam,ReferenceImage,newScaleArea)
% Creates a window which allows the user to get information about an
% object that is placed in the view of a camera.

% Setup MatConvNet
addpath ~/Documents/MATLAB/matconvnet-1.0-beta21/matlab
vl_setupnn;

% Load a model and upgrade it to MatConvNet current version.
disp('Loading model...');
net = load('cnn_model.mat');
net = vl_simplenn_tidy(net);
disp('Finished loading model...');

if nargin >= 3
    realScaleArea = newScaleArea;
else
    realScaleArea = 7000;  % in mm^2
end
threshold = [0,0,0;255,255,255];
lastArea = 0;
newRefPolygon = 0;

refImage = imread(ReferenceImage);
refImage = rgb2gray(refImage);

refPoints = detectSURFFeatures(refImage);
[refFeatures, refPoints] = extractFeatures(refImage, refPoints);

%display(length(boxFeatures));
if length(refFeatures) > 200
    refFeatures = refFeatures(1:200,:);
end

h = figure;
% UI controls
uicontrol('Style','pushbutton','Units','normalized','position',[0.0,0.0,0.1,0.05],'String','Identify','callback',@GetObjectArea);    % button to get area
uicontrol('Style','pushbutton','Units','normalized','position',[0.1,0.0,0.1,0.05],'String','Get Volume','callback',@GetObjectVolume);    % button to get volume
LRSlider = uicontrol('style','slider','Units','normalized','Position',[0.00,0.15,0.02,0.2],'min',0,'max',255','value',0,'callback',@Slide);     % min red slider
LGSlider = uicontrol('style','slider','Units','normalized','Position',[0.02,0.15,0.02,0.2],'min',0,'max',255','value',0,'callback',@Slide);     % min green slider
LBSlider = uicontrol('style','slider','Units','normalized','Position',[0.04,0.15,0.02,0.2],'min',0,'max',255','value',0,'callback',@Slide);     % min blue slider
URSlider = uicontrol('style','slider','Units','normalized','Position',[0.00,0.40,0.02,0.2],'min',0,'max',255','value',255,'callback',@Slide);     % max red slider
UGSlider = uicontrol('style','slider','Units','normalized','Position',[0.02,0.40,0.02,0.2],'min',0,'max',255','value',255,'callback',@Slide);   % max green slider
UBSlider = uicontrol('style','slider','Units','normalized','Position',[0.04,0.40,0.02,0.2],'min',0,'max',255','value',255,'callback',@Slide);   % max blue slider
uicontrol('Style','text','Units','normalized','position',[0.00,0.34,0.02,0.05],'String','R');   % Red label
uicontrol('Style','text','Units','normalized','position',[0.02,0.34,0.02,0.05],'String','G');   % Green label
uicontrol('Style','text','Units','normalized','position',[0.04,0.34,0.02,0.05],'String','B');   % Blue label
uicontrol('Style','text','Units','normalized','position',[0.07,0.5,0.05,0.03],'String','Max');  % Maximum label
uicontrol('Style','text','Units','normalized','position',[0.07,0.25,0.05,0.03],'String','Min'); % Minimum label

isRunning = true;
while ishandle(h)
    if isRunning
        sceneImage = snapshot(cam);
        figure(1);
        %% find scale
        sceneImage2 = rgb2gray(sceneImage);
        scenePoints = detectSURFFeatures(sceneImage2);
        
        [sceneFeatures, scenePoints] = extractFeatures(sceneImage2, scenePoints);
        
        refPairs = matchFeatures(refFeatures, sceneFeatures);
        if length(refPairs) > 3
            matchedrefPoints = refPoints(refPairs(:, 1), :);
            matchedScenePoints = scenePoints(refPairs(:, 2), :);            
            
            tform = estimateGeometricTransform(matchedrefPoints, matchedScenePoints, 'affine');
            
            refPolygon = [1, 1;...                           % top-left
                size(refImage, 2), 1;...                 % top-right
                size(refImage, 2), size(refImage, 1);... % bottom-right
                1, size(refImage, 1);...                 % bottom-left
                1, 1];                   % top-left again to close the polygon
            
            newRefPolygon = transformPointsForward(tform, refPolygon);
            
        end
        
        %% Image masking
        objectMask = sceneImage(:,:,1) < threshold(1,1) | sceneImage(:,:,2) < threshold(1,2) | sceneImage(:,:,3) < threshold(1,3)...
            | sceneImage(:,:,1) > threshold(2,1) | sceneImage(:,:,2) > threshold(2,2) | sceneImage(:,:,3) > threshold(2,3);
        
        %exclude the scale from the mask
        if length(refPairs) > 3
            
            scaleMask = roipoly(sceneImage(:,:,1),newRefPolygon(:,1),newRefPolygon(:,2));
            %combine masks.
            objectMask = objectMask+scaleMask;
            mask = objectMask & ~scaleMask;
        else
            mask = objectMask;
        end
        
        % make stuff filtered out white.
        for k = 1:3
            tmp = squeeze(sceneImage(:,:,k));
            tmp(mask) = 255;
            sceneImage(:,:,k) = tmp;
        end
        %% display image
        imshow(sceneImage);
        if length(refPairs) > 3
            
            hold on;
            line(newRefPolygon(:, 1), newRefPolygon(:, 2), 'Color', 'y');
            hold off;
        end
    end
end


%% callback functions
    function GetObjectArea(~,~)
        % Finds the rea life size of the object
        objectPixelCount = sum(~objectMask(:));
        scaleFactor = GetScaleArea();
        apparentArea = scaleFactor*objectPixelCount;
        lastArea = apparentArea;

        % pass img to kendrick's object identification program.
        img_conv = snapshot(cam);  

        im_ = single(img_conv) ; % note: 255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - net.meta.normalization.averageImage ;

        % Run the CNN.
        res = vl_simplenn(net, im_) ;

        % Show the classification result.
        figure(2);
        scores = squeeze(gather(res(end).x)) ;
        [bestScore, best] = max(scores) ;
        clf ; imagesc(img_conv);
        figure_text = sprintf('%s, %s mm^2, score %.3f', net.meta.classes.description{best}, apparentArea, bestScore);
        title(figure_text);

        isRunning = true;
    end

    function GetObjectVolume(~,~)
        % Pairs the height of the object with a previous area to get the
        % volume of the object.
        top = find(sum(~objectMask,1),~0,'first');
        bottom = find(sum(~objectMask,1),~0,'last');
        scaleFactor = GetScaleArea();
        pixelHeight = bottom-top+1;
        actualHeight = pixelHeight*sqrt(scaleFactor);
        volume = lastArea*actualHeight;
        display(volume);
    end

    function mmPerPixel = GetScaleArea(~,~)
        % Gets the scale factor of the image in mm^2 per pixel
        scaleArea = polyarea(newRefPolygon(:,1),newRefPolygon(:,2));
        mmPerPixel = realScaleArea/scaleArea;
        display(mmPerPixel);
    end

    function Slide(~,~)
        % Gets the values from the slider controls and assigns them to the
        % threshold
        threshold(1,1) = get(LRSlider,'value');
        threshold(1,2) = get(LGSlider,'value');
        threshold(1,3) = get(LBSlider,'value');
        threshold(2,1) = get(URSlider,'value');
        threshold(2,2) = get(UGSlider,'value');
        threshold(2,3) = get(UBSlider,'value');
    end

end
