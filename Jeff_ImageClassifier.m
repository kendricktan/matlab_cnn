function Jeff_ImageClassifier(cam,newScale)

% Setup MatConvNet
addpath ~/Documents/MATLAB/matconvnet-1.0-beta21/matlab
vl_setupnn;

% Load a model and upgrade it to MatConvNet current version.
disp('Loading model...');
net = load('imagenet-caffe-alex.mat');
net = vl_simplenn_tidy(net);
disp('Finished loading model...');

% Jeff's stuff
area = 0; % area for image pairing.
if nargin > 1
    scaleArea = newScale;
else
    scaleArea = pi*60^2;
end
NOISE_REDUC_SIZE = 10;
NOISE_REDUC_THRESHOLD = 30;
isRunning = true;
threshold = [0,0,0;0,255,255];
scaleThresh = [0,0,0;0,255,255];
RGBaxes = figure;
% ui controls.
convButton = uicontrol('style','pushbutton','position', [200,50,50,20],'string','Conv','callback',@GetObject);
button = uicontrol('style','pushbutton','position', [20,50,50,20],'string','Get area','callback',@GetArea);
button2 = uicontrol('style','pushbutton','position', [100,50,50,20],'string','Get volume','callback',@GetVolume);
typeToggle = uicontrol('style','pushbutton','position',[500,50,50,20],'string','Area','callback',@Toggle);
LRSlider = uicontrol('style','slider','Position',[20,200,10,100],'min',0,'max',255','value',0,'callback',@Slide);
LGSlider = uicontrol('style','slider','Position',[30,200,10,100],'min',0,'max',255','value',0,'callback',@Slide);
LBSlider = uicontrol('style','slider','Position',[40,200,10,100],'min',0,'max',255','value',0,'callback',@Slide);
URSlider = uicontrol('style','slider','Position',[510,200,10,100],'min',0,'max',255','value',0,'callback',@Slide);
UGSlider = uicontrol('style','slider','Position',[520,200,10,100],'min',0,'max',255','value',255,'callback',@Slide);
UBSlider = uicontrol('style','slider','Position',[530,200,10,100],'min',0,'max',255','value',255,'callback',@Slide);
SURSlider = uicontrol('style','slider','Position',[200,370,100,10],'min',0,'max',255','value',0,'callback',@sSlide);
SUGSlider = uicontrol('style','slider','Position',[200,360,100,10],'min',0,'max',255','value',255,'callback',@sSlide);
SUBSlider = uicontrol('style','slider','Position',[200,350,100,10],'min',0,'max',255','value',255,'callback',@sSlide);
SBRSlider = uicontrol('style','slider','Position',[100,370,100,10],'min',0,'max',255','value',0,'callback',@sSlide);
SBGSlider = uicontrol('style','slider','Position',[100,360,100,10],'min',0,'max',255','value',0,'callback',@sSlide);
SBBSlider = uicontrol('style','slider','Position',[100,350,100,10],'min',0,'max',255','value',0,'callback',@sSlide);
% video loop
while ishandle(RGBaxes)
    if isRunning                
        img = snapshot(cam);
        figure(1);
        % create masks for scale and object
        scaleMask = img(:,:,1) >= scaleThresh(1,1) & img(:,:,2) >= scaleThresh(1,2) & img(:,:,3) >= scaleThresh(1,3)...
            & img(:,:,1) <= scaleThresh(2,1) & img(:,:,2) <= scaleThresh(2,2) & img(:,:,3) <= scaleThresh(2,3);
        objectMask = img(:,:,1) >= threshold(1,1) & img(:,:,2) >= threshold(1,2) & img(:,:,3) >= threshold(1,3)...
            & img(:,:,1) <= threshold(2,1) & img(:,:,2) <= threshold(2,2) & img(:,:,3) <= threshold(2,3);
        % noise reduction.
        onesMatrix = ones(NOISE_REDUC_SIZE,NOISE_REDUC_SIZE);
        noiseMask = conv2(double(scaleMask),onesMatrix,'same') > NOISE_REDUC_THRESHOLD;
        scaleMask = scaleMask & noiseMask;
        noiseMask = conv2(double(objectMask),onesMatrix,'same') > NOISE_REDUC_THRESHOLD;
        objectMask = objectMask & noiseMask;

        % red mask for the object
        tmp = squeeze(img(:,:,1));
        tmp(objectMask) = 255;
        img(:,:,1) = tmp;
        for k = 2:3
            tmp = squeeze(img(:,:,k));
            tmp(objectMask) = 0;
            img(:,:,k) = tmp;
        end
        % blue mask for the scale.
        for k = 1:2
            tmp = squeeze(img(:,:,k));
            tmp(scaleMask) = 0;
            img(:,:,k) = tmp;
        end
        tmp = squeeze(img(:,:,3));
        tmp(scaleMask) = 255;
        img(:,:,3) = tmp;
        %limits
        if strcmp(typeToggle.String, 'Limits')
            top = find(sum(objectMask,2),~0,'first');
            bottom = find(sum(objectMask,2),~0,'last');
            left = find(sum(objectMask,1),~0,'first');
            right = find(sum(objectMask,1),~0,'last');
            img = insertShape(img,'rectangle',[left,top,right-left,bottom-top],'Color','black','lineWidth',5);
        end
        imshow(img);
    end
end

function objectArea = GetObjectArea()
    if strcmp(typeToggle.String,'Area')
        objectArea = sum(objectMask(:));
    else
        objectArea = (bottom-top+1)*(right-left+1);
    end
    display(['Object area: ', num2str(objectArea),' pixels']);
end

function mmPerPixel = GetScale(~,~)
    pixels = sum(scaleMask(:));
    mmPerPixel = scaleArea/pixels;
    display(['Scale: ', num2str(mmPerPixel),' mm per pixel']);
end

function GetArea(~,~)
    scale = GetScale();
    objectPixels = GetObjectArea();
    area = objectPixels*scale;
    display(['Area: ', num2str(area),' mm^2']);
end
function GetVolume(~,~)
    height = (bottom-top+1)*sqrt(GetScale());
    volume = height*area;
    display(['Volume: ', num2str(volume),' mm^3']);
end

function Slide(~,~)
    threshold(1,1) = get(LRSlider,'value');
    threshold(1,2) = get(LGSlider,'value');
    threshold(1,3) = get(LBSlider,'value');
    threshold(2,1) = get(URSlider,'value');
    threshold(2,2) = get(UGSlider,'value');
    threshold(2,3) = get(UBSlider,'value');
end
function sSlide(~,~)
    scaleThresh(1,1) = get(SBRSlider,'value');
    scaleThresh(1,2) = get(SBGSlider,'value');
    scaleThresh(1,3) = get(SBBSlider,'value');
    scaleThresh(2,1) = get(SURSlider,'value');
    scaleThresh(2,2) = get(SUGSlider,'value');
    scaleThresh(2,3) = get(SUBSlider,'value');
end
function Toggle(~,~)
    if strcmp(typeToggle.String,'Area')
        set(typeToggle,'String','Limits');
    else
        set(typeToggle,'String','Area');
    end
end
function GetObject(~,~)
    % do convolutions on the picture to detect type of object.
    isRunning = false;        
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
    clf ; imagesc(img_conv) ;
    figure_text = sprintf('%s, score %.3f', net.meta.classes.description{best}, bestScore)        
    title(figure_text);

    isRunning = true;
end

end