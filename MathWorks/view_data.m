clear;
close all; clc;

% seed = 12345;
% rng(seed);

%%
filename = '../data.csv';

fid = fopen(filename); 
T = textscan(fid, '%f,%f');
fclose(fid);

u = T{1};
y = T{2};

assert(length(u) == length(y));
N = length(u);

fig = figure('Name', 'Input-Output');
hold all;
stairs(10*u, 'DisplayName', '10*u');
stairs(y, 'DisplayName', 'y');
xlim([0, N+1]);
legend('show', 'Location', 'SouthEast');
grid on;

set(fig, 'PaperPositionMode', 'auto');
saveas(fig, '../img/input_output.png', 'png');

%%
z = tf('z', -1);

G_0 = (2*z^2 + 2*z - 1.5) / (z^3 - 1.4*z^2 + 0.48*z);
H_0 = (z^3)               / (z^3 - 1.4*z^2 + 0.48*z);

mu = 0;
sigma = sqrt(8);
epsilon = normrnd(mu, sigma, N, 1);

y_0 = lsim(G_0, u);
nu_0 = lsim(H_0, epsilon);

fig = figure('Name', 'Output vs. Simulation');
hold all;
% stairs(u, 'DisplayName', 'u');
stairs(y, 'DisplayName', 'y');
stairs(y_0 + nu_0, 'DisplayName', 'y_s');
xlim([0, N+1]);
legend('show', 'Location', 'SouthEast');
grid on;
