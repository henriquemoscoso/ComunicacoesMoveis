clear; clc; close all
rng(42);

%% Parâmetros
gbar_dB    = [-20 0 20];          % SNR média por símbolo (dB)
gth_teo_dB = -30:0.1:30;          % limiar, curva teórica (dB)
gth_sim_dB = -30:2:30;            % limiar, pontos simulados (dB)
K_fix      = [1 5];               % fatores de Rice fixos
Nt         = 1e9;                 % amostras por curva 
bloco      = 1e6;                 % amostras por bloco (controle de memória)>> 1/
OP_min     = 100/Nt;              % menor OP confiável
N_pts_min  = 3;                   % mínimo de pontos simulados na transição da curva
K_busca    = 6:500;               % candidatos para o K_R máximo

cores = [0 0 1; 1 0 0; 0 0.8 0];  % azul, vermelho, verde (como Figs. 10 e 11)

gth_teo = 10.^(gth_teo_dB/10);
gth_sim = 10.^(gth_sim_dB/10);
gbar    = 10.^(gbar_dB/10);

%% Busca do maior K_R viável
K_max = K_fix(end);
for K = K_busca
    ok = true;
    for j = 1:numel(gbar)
        p = op_rice(gth_sim, gbar(j), K);
        if sum(p >= OP_min & p <= 0.99) < N_pts_min
            ok = false; break;
        end
    end
    if ~ok, break; end
    K_max = K;
end
fprintf('K_R máximo viável (Nt = %.0e, passo de %g dB): %d\n', ...
        Nt, gth_sim_dB(2)-gth_sim_dB(1), K_max);
K_vec = [K_fix K_max];

%% Rayleigh 
OPt_ray = zeros(numel(gbar), numel(gth_teo));
OPs_ray = zeros(numel(gbar), numel(gth_sim));
for j = 1:numel(gbar)
    OPt_ray(j,:) = -expm1(-gth_teo/gbar(j));            
    OPs_ray(j,:) = simula_op(0, gbar(j), gth_sim, Nt, bloco);  % K = 0: Rayleigh
end
fprintf('Rayleigh concluído\n');

%% Rice 
for K = K_vec
    OPt_ric = zeros(numel(gbar), numel(gth_teo));
    OPs_ric = zeros(numel(gbar), numel(gth_sim));
    for j = 1:numel(gbar)
        OPt_ric(j,:) = op_rice(gth_teo, gbar(j), K);     
        OPs_ric(j,:) = simula_op(K, gbar(j), gth_sim, Nt, bloco);
    end
    fprintf('Rice K_R = %d concluído\n', K);

    figure('Color','w','Position',[100 100 760 460]);
    ax = axes; hold(ax,'on');
    h = gobjects(0); leg = {};

    for j = 1:numel(gbar)
        h(end+1) = plot(ax, gth_teo_dB, OPt_ray(j,:), '-', ...
                        'Color', cores(j,:), 'LineWidth', 1.5); 
        leg{end+1} = sprintf('Rayleigh - $\\bar{\\gamma}_s = %d$ dB', gbar_dB(j)); 
    end
    for j = 1:numel(gbar)
        h(end+1) = plot(ax, gth_teo_dB, OPt_ric(j,:), '--', ...
                        'Color', cores(j,:), 'LineWidth', 1.5); 
        leg{end+1} = sprintf('Rice - $\\bar{\\gamma}_s = %d$ dB', gbar_dB(j)); 
    end
    for j = 1:numel(gbar)
        plot(ax, gth_sim_dB, OPs_ray(j,:), 'k+', 'MarkerSize', 6, ...
             'LineWidth', 1.1, 'HandleVisibility', 'off');
        plot(ax, gth_sim_dB, OPs_ric(j,:), 'k+', 'MarkerSize', 6, ...
             'LineWidth', 1.1, 'HandleVisibility', 'off');
    end
    h(end+1) = plot(ax, nan, nan, 'k+', 'MarkerSize', 6, 'LineWidth', 1.1);
    leg{end+1} = 'Sim.';

    set(ax, 'YScale', 'log', 'TickLabelInterpreter', 'latex', 'FontSize', 11);
    grid(ax,'on'); grid(ax,'minor'); box(ax,'on');
    xlim(ax, [gth_teo_dB(1) gth_teo_dB(end)]);
    ylim(ax, [1e-6 1]);
    xlabel(ax, 'Limiar de SNR - $\gamma_{\mathrm{th}}$ (dB)', 'Interpreter', 'latex');
    ylabel(ax, 'OP', 'Interpreter', 'latex');
    title(ax, sprintf('Probabilidade de \\textit{Outage} - $K_R = %d$', K), ...
          'Interpreter', 'latex');
    legend(ax, h, leg, 'Interpreter', 'latex', 'Location', 'southeast');

   
end

%% 

function F = op_rice(gth, gbar, K)
a = sqrt(2*K);
F = zeros(size(gth));
f = @(t) t.*exp(-(t-a).^2/2).*besseli(0, a*t, 1);
for n = 1:numel(gth)
    b = sqrt(2*(K+1)*gth(n)/gbar);
    if b > a + 40                  % cauda desprezível
        F(n) = 1;
    elseif a > 0 && a < b          % ponto de pico dentro do intervalo
        F(n) = integral(f, 0, b, 'Waypoints', a, 'RelTol', 1e-10, 'AbsTol', 0);
    else
        F(n) = integral(f, 0, b, 'RelTol', 1e-10, 'AbsTol', 0);
    end
end
F = min(F, 1);
end

function OP = simula_op(K, gbar, gth, Nt, bloco)
% Monte Carlo da OP. h = componente LoS + componente difusa, E[|h|^2] = 1.

Ns = zeros(1, numel(gth));
for ini = 1:bloco:Nt
    n  = min(bloco, Nt-ini+1);
    h  = sqrt(K/(K+1)) + sqrt(1/(2*(K+1)))*(randn(n,1) + 1j*randn(n,1));
    gs = gbar*abs(h).^2;                 
    Ns = Ns + sum(gs <= gth(:).', 1);    
end
OP = Ns/Nt;                              
OP(Ns < 100) = NaN;
end