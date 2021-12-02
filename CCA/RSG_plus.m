function RSG_plus(data_choice, k)

% This code needs the 'manopt' package. It can be downloaded from here: https://www.manopt.org/downloads.html
addpath(genpath('path/to/manopt/')); 
warning off;


if strcmp(data_choice,'mnist')==1
%MNIST
dataname = 'mnist';
k = k;
d = 392;
d1 = d;
d2 = d;
N = 60000; %%Check if N is divisible by k
bsize = 100;
nbatch = N/bsize;%% nbatch * bsize = N

load('Datasets/MNIST/mnist.mat');
Z = double(trainX)/255;
X = Z(1:N, 1:392);
Y = Z(1:N,393:784);
for i = 1:392
    X(:,i) = X(:,i) - mean(X(:,i));
    Y(:,i) = Y(:,i) - mean(Y(:,i));
end

[gt_U,gt_V,r] = canoncorr(X, Y);



elseif strcmp(data_choice,'mediamill')==1 
%Mediamill

dataname = 'mediamill';
k = k;
d = 101;
d1 = 120;
d2 = 101;
N = 25800; %%Check if N is divisible by k
bsize = 100;
nbatch = N/bsize;%% nbatch * bsize = N

load('Datasets/Mediamill/mediamill_trainX.mat');
load('Datasets/Mediamill/mediamill_trainY.mat');
X = trainX(1:N, :);
Y = trainY(1:N, :);
for i = 1:d1
    X(:,i) = X(:,i) - mean(X(:,i));
end
for i = 1:d2
    Y(:,i) = Y(:,i) - mean(Y(:,i));
end

[gt_U,gt_V,r] = canoncorr(X, Y);



elseif strcmp(data_choice,'cifar')==1  
%CIFAR
dataname = 'cifar';
k = k;
d = 1536;
d1 = d;
d2 = d;
N = 60000; %%Check if N is divisible by k
bsize = 100;
nbatch = N/bsize;%% nbatch * bsize = N

load('Datasets/CIFAR/train_x.mat');
load('Datasets/CIFAR/train_y.mat');
load('Datasets/CIFAR/test_x.mat');
load('Datasets/CIFAR/test_y.mat');

trainX = double(cifar_train_x(1:50000,:))/255;
trainY= double(cifar_train_y(1:50000,:))/255;
testX = double(test_x(1:10000,:))/255;
testY= double(test_y(1:10000,:))/255;
trainX = [trainX; testX];
trainY = [trainY; testY];
for i = 1:1536
    trainX(:,i) = trainX(:,i) - mean(trainX(:,i));
    trainY(:,i) = trainY(:,i) - mean(trainY(:,i));
end
X = trainX;
Y = trainY;

[gt_U,gt_V,r] = canoncorr(X, Y);


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
whiten_X = gt_U'*X'*X*gt_U;
for i = 1:k
    gt_U(:,i) = gt_U(:,i) ./ sqrt(whiten_X(i,i));
end
whiten_Y = gt_V'*Y'*Y*gt_V;
for i = 1:k
    gt_V(:,i) = gt_V(:,i) ./ sqrt(whiten_Y(i,i));
end
gt_obj = trace(gt_U(:,1:k)'*X'*Y*gt_V(:,1:k))
gt_UXXU = gt_U(:,1:k)'*X'*X*gt_U(:,1:k)

Du = eye(k);
Dv = eye(k);

Tx = eye(k);
Ty = eye(k);


manifold1 = rotationsfactory(k);
manifold2_x = stiefelfactory(d1, k);
manifold3_x = grassmannfactory(d1, k);
manifold2_y = stiefelfactory(d2, k);
manifold3_y = grassmannfactory(d2, k);


Xn = reshape(X, [nbatch,bsize,d1]);
Yn = reshape(Y, [nbatch,bsize,d2]);
nXX = permute(reshape(X,[N/k,k,d1]),[3,2,1]);
nYY = permute(reshape(Y,[N/k,k,d2]),[3,2,1]);
lr = 1e-2;
lr1 = 1e-1;
wwhite = 0.1; 
diter = 30; 




%streaming pca start
lr_pca = 0.01;
[Qx,~] = qr(rand(d1,k), 0);
[Qy,~] = qr(rand(d2,k), 0);
for batch = 1 : nbatch
    XX = squeeze(Xn(batch,:,:));
    YY = squeeze(Yn(batch,:,:));
    
    %update streaming pca
    Qx = Qx + lr_pca*XX'*XX*Qx;
    [Qx,~] = qr(Qx, 0);
    Qy = Qy + lr_pca*YY'*YY*Qy;
    [Qy,~] = qr(Qy, 0);
end


TCC_list = [0];
iter_list = [0];
t_list = [0];
t_total = 0;
t_start = tic;

for iter = 1 : 1
    
    if mod(iter,diter)==0
        lr = lr*0.5;
        lr1 = lr1*0.5;
    end
    for batch = 1 : nbatch
        
        XX = squeeze(Xn(batch,:,:));
        YY = squeeze(Yn(batch,:,:));



        XXn = reshape(XX',[d1,k,bsize/k]);
        YYn = reshape(YY',[d2,k,bsize/k]);

        Vx = ( XX'*( - YY*Qy*Ty)*Tx');
        Vy = (-YY'*(XX*Qx*Tx )*Ty' );


        tx = zeros(d1,k);
        ty = zeros(d2,k);
        for j=1:size(XXn,3)
            q = squeeze(XXn(:,:,j)); 
            tx = tx - real(manifold3_x.invretr(Qx, q))/size(XXn,3);
            q = squeeze(YYn(:,:,j)); 
            ty = ty - real(manifold3_y.invretr(Qy, q))/size(XXn,3);
        end


        Vx = Vx + wwhite*tx;
        Vy = Vy + wwhite*ty;

        Vx = Vx - Qx*Vx'*Qx;
        Vy = Vy - Qy*Vy'*Qy;

        Vx = normc(real(Vx));
        Vy = normc(real(Vy));

        VTx = ( Qx'*XX'*( - YY*Qy*Ty));
        VTy = ( -Qy'*YY'*(XX*Qx*Tx ));


        Qx = manifold2_x.retr(Qx, -lr1*Vx);

        Qy = manifold2_y.retr(Qy, -lr1*Vy);



        VTx = (VTx - VTx');
        VTy = (VTy - VTy');
        VTx = normc(real(VTx));
        VTy = normc(real(VTy));

        VTx = 0.5*VTx;
        VTy = 0.5*VTy;
        
        Tx = Tx*manifold1.retr(eye(k), -lr*VTx);
        Ty = Ty*manifold1.retr(eye(k), -lr*VTy);
 
        
        if mod(batch*bsize, 1000)==0
        t_total = t_total + toc(t_start);
        
        Ax = Qx*Tx;
        Ay = Qy*Ty;
        nAx = Ax;
        nAy = Ay;
        idx = diag(nAx'*X'*Y*nAy)<0;
        Du = (1./sqrt(diag(nAx'*X'*X*nAx)));
        Dv = diag(1./sqrt(diag(nAy'*Y'*Y*nAy)));

        Du = diag(idx .* -Du + ~idx.*Du);
        Du'*nAx'*X'*X*nAx*Du;
        trace(real(Du'*nAx'*X'*Y*nAy*Dv));
        [diag(real(Du'*nAx'*X'*Y*nAy*Dv))' trace(real(Du'*nAx'*X'*Y*nAy*Dv))];

        X_our = X*nAx*Du;
        X_gt = X*gt_U(:,1:k);
        Y_our = Y*nAy*Dv;
        Y_gt = Y*gt_V(:,1:k);
        [~,~,r1] = canoncorr(X_our, Y_our);
        [~,~,r2] = canoncorr(X_gt, Y_gt);
        TCC=sum(r1)/sum(r2)
        
        iter_list = [iter_list, batch*bsize];
        TCC_list = [TCC_list, TCC];
        t_list = [t_list, t_total];
        t_start = tic;
        
        end
        
        
    end


    t_total = t_total + toc(t_start);
    Ax = Qx*Tx;
    Ay = Qy*Ty;
    nAx = Ax;
    nAy = Ay;
    idx = diag(nAx'*X'*Y*nAy)<0;
    Du = (1./sqrt(diag(nAx'*X'*X*nAx)));
    Dv = diag(1./sqrt(diag(nAy'*Y'*Y*nAy)));

    Du = diag(idx .* -Du + ~idx.*Du);
    Du'*nAx'*X'*X*nAx*Du;
    trace(real(Du'*nAx'*X'*Y*nAy*Dv));
    [diag(real(Du'*nAx'*X'*Y*nAy*Dv))' trace(real(Du'*nAx'*X'*Y*nAy*Dv))];

    X_our = X*nAx*Du;
    X_gt = X*gt_U(:,1:k);
    Y_our = Y*nAy*Dv;
    Y_gt = Y*gt_V(:,1:k);
    [~,~,r1] = canoncorr(X_our, Y_our);
    [~,~,r2] = canoncorr(X_gt, Y_gt);
    TCC=sum(r1)/sum(r2)

    iter_list = [iter_list, batch*bsize];
    TCC_list = [TCC_list, TCC];
    t_list = [t_list, t_total];
end

save(strcat('saved_mat_files/our_TCC_',strcat(dataname,int2str(k))), 'TCC_list');
save(strcat('saved_mat_files/our_iter_',strcat(dataname,int2str(k))), 'iter_list');
save(strcat('saved_mat_files/our_t_',strcat(dataname,int2str(k))), 't_list');

clear all;
