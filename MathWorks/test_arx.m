clear;
close all; clc;

z = zpk('z');

G = (z - 0.8) / (z - 0.9) / (z - 0.4);

k = (0:1:20)';

u = sin(0.5*k) + sin(0.25*k);

y = lsim(G, u, k);

figure;
hold all;
stairs(k, u);
stairs(k, y);

G_arx = arx([y u], [2, 2, 1]);
G_arx = zpk(G_arx);
G_arx.var = 'z'