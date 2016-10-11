function varargout = custom_gui(varargin)
% CUSTOM_GUI MATLAB code for custom_gui.fig
%      CUSTOM_GUI, by itself, creates a new CUSTOM_GUI or raises the existing
%      singleton*.
%
%      H = CUSTOM_GUI returns the handle to a new CUSTOM_GUI or the handle to
%      the existing singleton*.
%
%      CUSTOM_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CUSTOM_GUI.M with the given input arguments.
%
%      CUSTOM_GUI('Property','Value',...) creates a new CUSTOM_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before custom_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to custom_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help custom_gui

% Last Modified by GUIDE v2.5 12-Oct-2016 00:04:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @custom_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @custom_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before custom_gui is made visible.
function custom_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to custom_gui (see VARARGIN)

% Choose default command line output for custom_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes custom_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = custom_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pbIdentify.
function pbIdentify_Callback(hObject, eventdata, handles)
% hObject    handle to pbIdentify (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Read and preprocess image
global vid
global net
im = getsnapshot(vid);
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


% --- Executes on button press in pbPreview.
function pbPreview_Callback(hObject, eventdata, handles)
% hObject    handle to pbPreview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% choose which webcam (winvideo-1) and which  mode (YUY2_176x144)
% Setup MatConvNet
addpath ~/Documents/MATLAB/matconvnet-1.0-beta21/matlab
vl_setupnn;

% Load a model and upgrade it to MatConvNet current version.
disp('Loading model...');
global net
net = load('imagenet-caffe-alex.mat');
net = vl_simplenn_tidy(net);
disp('Finished loading model...');

% Video capture
global vid
vid = videoinput('linuxvideo');
% only capture one frame per trigger, we are not recording a video
vid.FramesPerTrigger = 1;
% output would image in RGB color space
vid.ReturnedColorspace = 'rgb';
% tell matlab to start the webcam on user request, not automatically
triggerconfig(vid, 'manual');
% we need this to know the image height and width
vidRes = get(vid, 'VideoResolution');
% image width
imWidth = vidRes(1);
% image height
imHeight = vidRes(2);
% number of bands of our image (should be 3 because it's RGB)
nBands = get(vid, 'NumberOfBands');
% create an empty image container and show it on axPreview
hImage = image(zeros(imHeight, imWidth, nBands), 'parent', handles.axPreview);
% begin the webcam preview
preview(vid, hImage);
