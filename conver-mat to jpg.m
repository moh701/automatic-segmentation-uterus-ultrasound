% Load the MAT-file
matFilePath = 'Z:\Uterine_segmentation\HealthyVolunteers\Delineations\UV038\endometrium.mat';
loadedData = load(matFilePath);

% Check if the necessary variables exist in the loaded data
if isfield(loadedData, 'BW') && isfield(loadedData, 'xi2') && isfield(loadedData, 'yi2')
    BW = loadedData.BW;
    xi2 = loadedData.xi2;
    yi2 = loadedData.yi2;
    
    % Display the binary mask (BW) and save it as a JPG
    figure;
    imshow(BW, []);
    title('Binary Mask (BW)');
    imwrite(BW, 'BW_mask.jpg');  % Save the BW image as a JPG
    
    % Create a figure to plot the delineation
    figure;
    imshow(BW, []);
    hold on;
    plot(xi2, yi2, 'r-', 'LineWidth', 2);  % Plot the delineation
    title('Delineation');
    hold off;
    saveas(gcf, 'Delineation.jpg');  % Save the delineation plot as a JPG
else
    error('The required variables BW, xi2, and yi2 are not present in the loaded MAT file.');
end

% If there are additional images in the MAT file, save them as JPGs
imageFields = fieldnames(loadedData);
for i = 1:numel(imageFields)
    fieldName = imageFields{i};
    if ~ismember(fieldName, {'BW', 'xi2', 'yi2'}) && ndims(loadedData.(fieldName)) == 2
        % If the field is an image (assuming 2D matrix), save it as a JPG
        figure;
        imshow(loadedData.(fieldName), []);
        title(fieldName);
        imwrite(loadedData.(fieldName), strcat(fieldName, '.jpg'));  % Save the image as a JPG
    end
end
