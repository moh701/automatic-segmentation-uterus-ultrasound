
clc; close all

% load the dicom data and info
datapath = '\\tue033068.ele.tue.nl\Uterine-CEUS\Uterine_segmentation\PATIENT_DATA\';
datapathdelin = '\\tue033068.ele.tue.nl\Uterine-CEUS\Uterine_segmentation\Delineations\';
ptname = 'UV038\';
fname = 'UV038_0019';
% roiname = 'uterus';
% roiname = 'hyper';
roiname = 'endometrium';
% roiname = 'endometrium';
info = dicominfo(strcat(datapath,ptname,fname));
totalframes = info.NumberOfFrames;


% files:
% pt 4 (1): CEUSPILOT004_0011
% pt 4 (2): CEUSPILOT004_0012
% pt 13: CEUSPILOT013_0015
% pt 14: IMG_20230214171934_0021
% pt 17: IMG_20230315085913_0138
% pt 18: IMG_20230425140822_0016
% pt 20: IMG_20230426150438_0016
% pt 21: IMG_20230516171505_0024
% pt 26: IMG_20230531150408_0019
% pt 31: IMG_20230626214817_0049

data = dicomread(strcat(datapath,ptname,fname),"frames",(50:50:totalframes));

%
% initiate variables
frame = 3; % equivalent to frame = 150

% select image of interes
Cbox_CEUS = [info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0,...
    info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0,...
    info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1,...
    info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1]+1;

CEUSimg_ori = data(Cbox_CEUS(2):Cbox_CEUS(4),Cbox_CEUS(1):Cbox_CEUS(3),:,:);
CEUSimg = squeeze(CEUSimg_ori(:,:,:,frame));

Cbox_Bmode = [info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMinX0,...
    info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMinY0,...
    info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMaxX1,...
    info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMaxY1]+1;

Bmodeimg = data(Cbox_Bmode(2):Cbox_Bmode(4),Cbox_Bmode(1):Cbox_Bmode(3),:,:);
Bmodeimg = squeeze(Bmodeimg(:,:,:,frame));

% figure(); imagesc(Bmodeimg)
% figure(); imagesc(CEUSimg)


%%
% delineation
figure;
[BW,xi2,yi2] = roipoly(Bmodeimg);

% %
% % Apply delineation to B-mode delineation
% imagesc(Bmodeimg)
% colormap('gray')
% axis off 
% hold on;
% title('B-mode')
% contour(BW,'LineColor','r');
% 
% % Apply B-mode delineation to CEUS
% CEUSimg = squeeze(CEUSimg);
% imagesc(CEUSimg)
% colormap('gray')
% axis off 
% hold on;
% title('CEUS')
% contour(BW,'LineColor','r');
%%
% 
% myVideo = VideoWriter(strcat(datapath,'Delineation_CEUSPILOT_011')); % To recording a video
% myVideo.FrameRate = 20;                  % To recording a video
% open(myVideo);                           % To recording a video
% 
% for k = 1:size(CEUSimg_ori,4)
%     imagesc(CEUSimg_ori(:,:,:,k)); hold on;
%     contour(BW,'LineColor','r');
%     hold on
%     pause(0.01);
%     frame = getframe(gcf);               % To recording a video
%     writeVideo(myVideo, frame);          % To recording a video
% end
% close(myVideo) 

%% 

save(strcat(datapathdelin,ptname,roiname),"BW","xi2","yi2");
