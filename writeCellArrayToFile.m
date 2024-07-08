function writeCellArrayToFile(cellArray, filename)
    % Open the text file for writing
    fileID = fopen(filename, 'w');
    
    if fileID == -1
        error('Error opening the file.');
    end
    
    % Iterate over the cell array and write each element to the file
    for i = 1:numel(cellArray)
        element = cellArray{i};
        
        % Check the type of element and write accordingly
        if isnumeric(element) && ismatrix(element)
            % Write matrices and numeric arrays
            [rows, cols] = size(element);
            fprintf(fileID, 'Element %d (Matrix %dx%d):\n', i, rows, cols);
            formatSpec = [repmat('%f ', 1, cols) '\n'];
            fprintf(fileID, formatSpec, element');
        elseif isnumeric(element)
            % Write numeric scalars
            fprintf(fileID, 'Element %d (Float): %f\n', i, element);
        elseif ischar(element)
            % Write character arrays (strings)
            fprintf(fileID, 'Element %d (String): %s\n', i, element);
        else
            fprintf(fileID, 'Element %d (Unknown Type): Unable to write\n', i);
        end
        
        % Add a blank line for readability
        fprintf(fileID, '\n');
    end
    
    % Close the file
    fclose(fileID);
end
