function convnet = cnn_model()
    % Gets an architecture for the convolutional neural network
    % Based off AlexNet
    
    % Inputs
    input = imageInputLayer([227 227 3]);
    
    % Convolution layers
    conv1 = convolution2dLayer(11, 96, 'Stride', 4, 'Padding', 0);
    conv2 = convolution2dLayer(5, 256, 'Stride', 1, 'Padding', 2);
    conv3 = convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1);
    conv4 = convolution2dLayer(3, 256, 'Stride', 1, 'Padding', 1);  
    
    relu = reluLayer();
    pool = maxPooling2dLayer(3, 'Stride', 2);
    norm = crossChannelNormalizationLayer(5);  
    
    % Fully connected layer
    fcl1 = fullyConnectedLayer(4096);
    fcl2 = fullyConnectedLayer(1000);
    
    prob = softmaxLayer();
    
    % Classification
    output = classificationLayer();
    
    % Convolutional layer
    convnet = [
        input;
        conv1;
        relu;        
        norm;
        pool;
        conv2;
        relu;        
        norm;
        pool;
        conv3;
        relu;
        conv3;
        relu;
        conv4;
        relu;
        pool;
        norm;
        fcl1;
        relu;
        fcl1;
        relu;
        fcl2;
        prob;
        output;
    ];
end

