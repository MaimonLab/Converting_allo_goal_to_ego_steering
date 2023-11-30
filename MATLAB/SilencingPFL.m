clear all

% Define PB and FB angles
thetaRight = -[67.5 112.5 157.5 157.5 -157.5 -112.5 -112.5 ...
                -67.5 -22.5 -22.5 22.5 67.5];
thetaLeft = -[-67.5 -22.5 22.5 22.5 67.5 112.5 112.5 ...
                157.5 -157.5 -157.5 -112.5 -67.5];
thetaFB = -((0:30:330) + 15);

% Parameters from data fit
a = 29.2282;
b = 2.1736;
c = -0.7011;
d = 0.6299;

noise = 47;

% thetaGH is a range of heading and goal angles
H = -180:1:180;
H(end) = [];
G = [-135 -90 -45 45 90 135];

nTrial = 5000;
angErr = zeros(25,nTrial,6);

for nOut=0:24
    nOut
    for kk = 1:nTrial
        % Left and Right PFL3 rates
        %   indexed by (heading,goal,column)
        rML = zeros(length(H),length(G),12);
        rMR = zeros(length(H),length(G),12);

        for j=1:length(G)
                thetaN = G(j) + noise*randn(1,1);
            for i=1:12
                gSig = d*cos(pi*(thetaN-thetaFB(i))/180);
                hSig = cos(pi*(H-thetaLeft(i))/180);
                rML(:,j,i) = a*log(1+exp(b*(hSig+gSig+c)));
                hSig = cos(pi*(H-thetaRight(i))/180);
                rMR(:,j,i) = a*log(1+exp(b*(hSig+gSig+c)));
            end
        end

        if (nOut>0)  
            iFull = randperm(24);
            iOut = iFull(1:nOut);
            for i=1:nOut
                if (iOut(i)<13)
                    rML(:,:,iOut(i)) = 0;
                else
                    rMR(:,:,iOut(i)-12) = 0;
                end
            end
        end
        % Left and right LAL outputs and turning signal
        rLLAL = sum(rML,3);
        rRLAL = sum(rMR,3);
        rTurn = rRLAL-rLLAL;

% This computes where the turning signal=0 for all goal angles
% There are two zero crossing, this picks the stable one

        thetaC = 360*rand(size(G));
        for i=1:length(G)
            rPlus = rTurn(2:end,i)+eps;
            rMinus = rTurn(1:end-1,i)+eps;
            iS = find(rPlus.*rMinus<0);
            if (length(iS)>1)
                iW = find(rTurn(iS,i)>=0);
                if (length(iW)==1)
                    thetaC(i) = H(iS(iW)); 
                else
                    iPick = randi([1 length(iW)],1,1);
                    thetaC(i) = H(iS(iW(iPick))); 
                end
            elseif (isempty(iS))
                thetaC(i) = randi([-180 180]);
            end
        end
        testErr = abs(thetaC-G);
        testErr = testErr - 360*(testErr>180) + 360*(testErr<-180);
        testErr = testErr - 360*(testErr>180) + 360*(testErr<-180);
        testErr = testErr - 360*(testErr>180) + 360*(testErr<-180);
        testErr = testErr - 360*(testErr>180) + 360*(testErr<-180);
        angErr(nOut+1,kk,:) = abs(testErr);
    end
end

err = zeros(25,1);
for i=1:length(err)
    tT = squeeze(angErr(i,:,:));
    x = mean(cos(pi*tT(:)/180));
    y = mean(sin(pi*tT(:)/180));
    err(i) = 180*atan2(y,x)/pi;
end

NCF = zeros(25,nTrial);
for i=1:25
    tT = squeeze(angErr(i,:,:));
    NCF(i,:) = sum(abs(tT')<30);
end
NC = mean(NCF,2);

figure(1)
plot(0:24,err,'bo','linewidth',2)
hold on
plot(0:24,err,'b','linewidth',2)
hold off
xlim([0 24]);
ylim([00 100]);
xlabel('number of silenced PFL3s')
ylabel('mean absolute error (deg)')
box off

figure(2)
plot(0:24,NC,'bo','linewidth',2)
hold on
plot(0:24,NC,'b','linewidth',2)
hold off
xlim([0 24]);
ylim([0 4]);
xlabel('number of silenced PFL3s')
ylabel('mean number correct')
box off