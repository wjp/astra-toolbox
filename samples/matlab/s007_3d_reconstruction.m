% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
%            2014-2016, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@uantwerpen.be
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------
vol_geom = astra_create_vol_geom(128, 128, 128);

angles = linspace2(0, 2*pi, 180);
proj_geom = astra_create_proj_geom('cone', 1.0, 1.0, 128, 192, angles, 512, 0);

% Create a simple hollow cube phantom
cube = zeros(128,128,128);
cube(17:112,17:112,17:112) = 1;
cube(33:96,33:96,33:96) = 0;

cfg = astra_struct('cuda3d');
cfg.ProjectionKernel = 'line';
cfg.ProjectionGeometry = proj_geom;
cfg.VolumeGeometry = vol_geom;
projector_id = astra_mex_projector3d('create', cfg);

proj_id = astra_mex_data3d('create', '-proj3d', proj_geom, 0);
vol_id = astra_mex_data3d('create', '-vol', vol_geom, cube);

cfg = astra_struct('FP3D_CUDA');
cfg.ProjectorId = projector_id;
cfg.ProjectionDataId = proj_id;
cfg.VolumeDataId = vol_id;

alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('run', alg_id);
astra_mex_algorithm('delete', alg_id);

proj_data = astra_mex_data3d('get_single', proj_id);

[proj_id2, proj_data2] = astra_create_sino3d_cuda(cube, proj_geom, vol_geom);

i = 22;

% Display a single projection image
figure, imshow(squeeze(proj_data(:,i,:))',[])
figure, imshow(squeeze(proj_data2(:,i,:))',[])
figure, imshow(squeeze(proj_data(:,i,:) - proj_data2(:,i,:))',[])

%{
% Create a data object for the reconstruction
rec_id = astra_mex_data3d('create', '-vol', vol_geom);

% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('SIRT3D_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = proj_id;
cfg.ProjectorId = projector_id;


% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);

% Run 150 iterations of the algorithm
% Note that this requires about 750MB of GPU memory, and has a runtime
% in the order of 10 seconds.
tic
astra_mex_algorithm('iterate', alg_id, 150);
toc
% Get the result
rec = astra_mex_data3d('get', rec_id);
%}

W = opTomo('cuda', proj_geom, vol_geom);

tic
rec = lsqr(W, proj_data(:), 1e-4, 100);
toc
rec = reshape(rec, astra_geom_size(vol_geom));

figure, imshow(squeeze(rec(:,:,65)),[]);


% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
%astra_mex_algorithm('delete', alg_id);
astra_mex_projector3d('delete', projector_id);
%astra_mex_data3d('delete', rec_id);
astra_mex_data3d('delete', proj_id);
astra_mex_data3d('delete', vol_id);
