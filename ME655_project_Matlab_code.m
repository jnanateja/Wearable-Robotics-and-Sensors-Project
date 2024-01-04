% ME655-WS(BME656-WS) - Virtual Lab Experience 3
% last rev. Oct 2022

clear all
close all
clc

% Create Simulation data
load('ME655_Lect09_VL3_Data.mat')
LH_hip_traj = timeseries([Lhip_pos Lhip_vel Lhip_acc Rhip_pos Rhip_vel Rhip_acc],t);

% initial conditions for human controller
Lhip_pos0=Lhip_pos(1);
Lhip_vel0=Lhip_vel(1);
Tau_h_max=30; %[Nm] max torque human can exert

%% Robot motor control

% Oscillator and filter settings:
eps=12;
nu=.5;
M=6;
lambda=.95;
N=80;
h=2.5*N;

t_start=5.0; %[s] time of AFO activation
dt_active_control=10; %[s] wait 10s after AFO is active before turning assistance ON

%% CLME Training

% Select train dataset
train_indices = find(t <= 30);
Lhip_pos_train = Lhip_pos(train_indices);
Lhip_vel_train = Lhip_vel(train_indices);
Rhip_pos_train = Rhip_pos(train_indices);
Rhip_vel_train = Rhip_vel(train_indices);

% Normalize data
[xp, xh, LhipAvg, RhipAvg, Sp, Sh] = normalize_data(Lhip_pos_train, Lhip_vel_train, Rhip_pos_train, Rhip_vel_train);

% Calculate Mhp and Mhh
Mhp = xh' * xp / (length(xp) - 1);
Mhh = xh' * xh / (length(xh) - 1);

% Calculate C, K, and k
C = (Mhh \ Mhp)';
K = Sp * C * inv(Sh);
k = -K * RhipAvg' + LhipAvg';

% Given K and k, estimate from BLUE
Rhip_train = [Rhip_pos_train, Rhip_vel_train];
Lhip_pred = (K * Rhip_train' + k)';

%% Kalman Filter

T = t(2) - t(1); % Sample period
A = [1 T; 0 1]; G = [T^2/2; T]; % Process parameters
Q = G * G' * var(Lhip_acc(train_indices)); % Process noise covariances
H = eye(2); % Measure state directly
R = diag([var(Lhip_pred(:, 1) - Lhip_pos_train), var(Lhip_pred(:, 2) - Lhip_vel_train)]); % Measurement noise covariance

% [Rhip_pos_des, Rhip_vel_des] = LH_hip_traj(4,5);

function [Xp, Xh, LhipAvg, RhipAvg, Sp, Sh] = normalize_data(Lhip_pos, Lhip_vel, Rhip_pos, Rhip_vel)

    function [avg, z_score] = calculate_avg_and_zscore(data)
        avg = mean(data);
        z_score = (data - avg) / std(data);
    end

    [LhipAvg, z_scoreLhip] = calculate_avg_and_zscore([Lhip_pos, Lhip_vel]);
    [RhipAvg, z_scoreRhip] = calculate_avg_and_zscore([Rhip_pos, Rhip_vel]);

    % State normalization for Lhip
    Sp = eye(2); % Identity matrix for Z-score normalization
    Xp = z_scoreLhip;

    % State normalization for Rhip
    Sh = eye(2); % Identity matrix for Z-score normalization
    Xh = z_scoreRhip;
end
