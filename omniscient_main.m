% omniscient_main.m
% Project Omniscient - Python AI Backend Integration (With Logging)
try
    clear; clc; close all;
    
    % Initialize Logger
    logID = fopen('matlab_log.txt', 'w');
    fprintf(logID, '%s - Starting MATLAB GUI\n', datestr(now, 'HH:MM:SS'));
    
    disp('Waiting for Python Eye Tracker...');
    fprintf(logID, '%s - Opening UDP Port 5005\n', datestr(now, 'HH:MM:SS'));
    
    try
        u = udpport('LocalPort', 5005);
        fprintf(logID, '%s - UDP Port opened successfully\n', datestr(now, 'HH:MM:SS'));
    catch portErr
        fprintf(logID, '%s - ERROR opening UDP: %s\n', datestr(now, 'HH:MM:SS'), portErr.message);
        error('Failed to open UDP port');
    end
    
    % SETUP TESTING UI
    hFig = figure('Name', 'Omniscient - Testing UI', 'Position', [100, 100, 800, 600], 'MenuBar', 'none');
    hAx = axes('Parent', hFig, 'XLim', [0 800], 'YLim', [0 600], 'YDir', 'reverse');
    hold(hAx, 'on');
    hPointer = plot(hAx, 400, 300, 'ro', 'MarkerSize', 30, 'MarkerFaceColor', 'r');
    hText = text(hAx, 400, 100, 'Start the Python script!', 'FontSize', 32, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', 'b');
    
    smoothX = 400; smoothY = 300;
    firstPacketReceived = false;
    
    disp('Omniscient Testing UI is active!');
    
    while isgraphics(hFig)
        if u.NumBytesAvailable > 0
            if ~firstPacketReceived
                fprintf(logID, '%s - First UDP bytes received!\n', datestr(now, 'HH:MM:SS'));
                firstPacketReceived = true;
                set(hText, 'String', 'Tracking...');
            end
            
            % Read buffer
            packet = read(u, u.NumBytesAvailable, "char");
            jsonStr = string(packet);
            
            % Split by newline to get distinct JSON objects
            parts = split(jsonStr, newline);
            
            % Grab the last valid JSON string (to prevent parsing cut-off data)
            lastJson = "";
            for k = length(parts):-1:1
                if strlength(strtrim(parts(k))) > 5
                    lastJson = strtrim(parts(k));
                    break;
                end
            end
            
            if strlength(lastJson) > 0
                try
                    data = jsondecode(lastJson);
                    
                    targetX = max(0, min(1, (data.x - 0.4) / 0.2)) * 800;
                    targetY = max(0, min(1, (data.y - 0.3) / 0.4)) * 600;
                    
                    smoothX = 0.85 * smoothX + 0.15 * targetX;
                    smoothY = 0.85 * smoothY + 0.15 * targetY;
                    set(hPointer, 'XData', smoothX, 'YData', smoothY);
                    
                    currentStatus = 'Tracking...';
                    pointerColor = 'r';
                    
                    if data.lb && ~data.rb
                        currentStatus = 'LEFT CLICK!';
                        pointerColor = 'g';
                    elseif data.rb && ~data.lb
                        currentStatus = 'RIGHT CLICK!';
                        pointerColor = 'y';
                    end
                    
                    set(hText, 'String', currentStatus);
                    set(hPointer, 'MarkerFaceColor', pointerColor);
                catch jsonErr
                    fprintf(logID, '%s - JSON Parse Error: %s | Data: %s\n', datestr(now, 'HH:MM:SS'), jsonErr.message, lastJson);
                end
            end
        end
        drawnow limitrate;
    end
    
    fprintf(logID, '%s - Window closed. Exiting.\n', datestr(now, 'HH:MM:SS'));
    fclose(logID);
    clear u;
catch ME
    if exist('logID', 'var')
        fprintf(logID, 'FATAL ERROR: %s\n', ME.message);
        fclose(logID);
    end
    fid = fopen('omniscient_error.txt', 'w'); fprintf(fid, 'ERROR: %s\n', ME.message); fclose(fid);
    try clear u; catch; end; try close all; catch; end; rethrow(ME);
end
