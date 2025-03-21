% Define the paths for patients' DICOM files
patients = {
    'CEUSPILOT030', 'Z:\Uterine_segmentation\PATIENT_DATA\CEUSPILOT030\CEUSPILOT030_0021';
    'UV017','Z:\Uterine_segmentation\PATIENT_DATA\UV017\UV017_0021';
    'UV031', 'Z:\Uterine_segmentation\PATIENT_DATA\UV031\UV031_0017';
    'UV034', 'Z:\Uterine_segmentation\PATIENT_DATA\UV034\UV_0021';
    'UV036', 'Z:\Uterine_segmentation\PATIENT_DATA\UV036\UV036_0023';
    'UV038', 'Z:\Uterine_segmentation\PATIENT_DATA\UV038\UV038_0019';
   
};

% Base directory to save the results
base_save_dir = 'Z:\Uterine_segmentation\HealthyVolunteers';

% Process each patient
for i = 1:size(patients, 1)
    patient_id = patients{i, 1};
    dicom_path = patients{i, 2};

    % Load the DICOM file
    if exist(dicom_path, 'file') == 2
        info = dicominfo(dicom_path);
        image = dicomread(dicom_path);
        totalframes = info.NumberOfFrames;
    else
        warning('DICOM file does not exist for patient %s', patient_id);
        continue;
    end

    % Define paths to save the JPG files
    save_path_bmode = fullfile(base_save_dir, 'Bmode', patient_id);
    save_path_ceus = fullfile(base_save_dir, 'CEUS', patient_id);
    if ~exist(save_path_bmode, 'dir')
        mkdir(save_path_bmode);
    end
    if ~exist(save_path_ceus, 'dir')
        mkdir(save_path_ceus);
    end

    % Process and save each frame
    for frame = 1:totalframes
        % Extract the frames of interest
        Cbox_CEUS = [info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0, ...
                     info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0, ...
                     info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1, ...
                     info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1] + 1;

        Cbox_Bmode = [info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMinX0, ...
                      info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMinY0, ...
                      info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMaxX1, ...
                      info.SequenceOfUltrasoundRegions.Item_2.RegionLocationMaxY1] + 1;

        CEUSimg = image(Cbox_CEUS(2):Cbox_CEUS(4), Cbox_CEUS(1):Cbox_CEUS(3), :, frame);
        Bmodeimg = image(Cbox_Bmode(2):Cbox_Bmode(4), Cbox_Bmode(1):Cbox_Bmode(3), :, frame);

        % Check if images are empty
        if isempty(Bmodeimg) || isempty(CEUSimg)
            warning('Empty image for patient %s, frame %d', patient_id, frame);
            continue;
        end

        % Save B-mode image as JPG
        imwrite(Bmodeimg, fullfile(save_path_bmode, sprintf('Bmode_frame_%d.jpg', frame)));

        % Save CEUS image as JPG
        imwrite(CEUSimg, fullfile(save_path_ceus, sprintf('CEUS_frame_%d.jpg', frame)));
    end
end
