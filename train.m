%[xTrainImages,yTrainLabel,xTestImages,yTestLabels] = extract_cifar10();

load('trainData.mat');
load('testData.mat');
xTrainImages = trainImages;
yTrainLabel = trainLabels;
xTestImages = testImages;
yTestLabels = testLabels;
rng('default');

hiddenSize1 = 200; 
% set the number of hidden nodes in Layer 1
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
'MaxEpochs',400, ...
'L2WeightRegularization',0.004, ...
'SparsityRegularization',4, ...
'SparsityProportion',0.15, ...
'ScaleData', false);
feat1 = encode(autoenc1,xTrainImages);

hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
'MaxEpochs',400, ...
'L2WeightRegularization',0.002, ...
'SparsityRegularization',4, ...
'SparsityProportion',0.1, ...
'ScaleData', false);
feat2 = encode(autoenc2,feat1);

hiddenSize3 = 50;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
'MaxEpochs',400, ...
'L2WeightRegularization',0.002, ...
'SparsityRegularization',4, ...
'SparsityProportion',0.1, ...
'ScaleData', false);
feat3 = encode(autoenc3,feat2);

softnet1 = trainSoftmaxLayer(feat2,yTrainLabel,'MaxEpochs',100);
deepnet1 = stack(autoenc1,autoenc2,softnet1);

softnet2 = trainSoftmaxLayer(feat3,yTrainLabel,'MaxEpochs',100);
deepnet2 = stack(autoenc1,autoenc2,autoenc3,softnet2);

view(deepnet1)
view(deepnet2)

% %train without fine-tuning
% y = deepnet(xTestImages); 
% plotconfusion(yTestLabels,y);

% Perform fine tuning
deepnet1 = train(deepnet1,xTrainImages,yTrainLabel,'showResources','yes');
y = deepnet1(xTestImages,'showResources','yes'); 
figure('Name','network 1');
plotconfusion(yTestLabels,y);

deepnet2 = train(deepnet2,xTrainImages,yTrainLabel,'showResources','yes');
y = deepnet2(xTestImages,'showResources','yes'); 
figure('Name','network 2');
plotconfusion(yTestLabels,y);