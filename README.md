# 3D-Reconstruction using Plane Stereo and Multi-View Stereo

This is an implementation of stereo reconstruction and multi view stereo (plane sweep) algorithm from scratch. 

Two view stereo steps
- Rectify the views for simplygying the epipolar search line
- Compute disparity map using ssd,sad and zncc kernels
- Added LR consistency check for handling occlusion
- Reconstruction of the scene using disparity map and multi-pair aggregation

