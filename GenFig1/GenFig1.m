clear;
close all;
clc;

% seeds the random number generator
seed = rng(1234);

%% Start the parallel pool
parallel_nodes = 4;
if isempty(gcp('nocreate'))
    parpool(parallel_nodes, 'IdleTimeout', 240);
end

%% Generate N observations with noise variance sigma^2

L_vec = [100, 200, 400]; %different signal length
length_L = length(L_vec);
length_alpha = 30;
alpha_vec = linspace(1, 6, length_alpha);
num_iter = 100; % number of iterations per alpha
relerr_em = zeros(length_L, length_alpha, num_iter);
a = 100; % a scaling parameter for the number of measurements
for l = 1:length_L
    L = L_vec(l); %signal length
    N = round(a*L/log(L)); %number of measurements
    x_true = randn(L, 1);
    for n = 1:length_alpha
        alpha = alpha_vec(n);
        sigma = sqrt(L/alpha/log(L)); %noise level
        for iter = 1:num_iter
            X_data = generate_observations(x_true, N, sigma);
            tic
            x_em = MRA_EM(X_data, sigma); %run EM
            em_time = toc;
            relerr_em(l, n, iter) = relative_error(x_true, x_em); % 2-norm, up to integer shifts
            fprintf('L = %g, alpha = %g, iter = %.3g, err_em = %.2g\n', L, alpha, iter, relerr_em(l, n, iter));
            fprintf('EM time = %g [sec]\n', em_time);
        end
        clear X_data
        save('XP_em.mat');
    end
end

%% plotting

mean_err_em = mean(relerr_em, 3); %average error
ln = 1.3;
figure;
hold on;
for i = 1:3
    plot(alpha_vec, mean_err_em(i,:),'linewidth', ln);
end
plot(alpha_vec, 1./sqrt(1 + a*alpha_vec),'linewidth', ln); %expected error if the shifts were known
xline(2,'--r');
legend('L = 100', 'L = 200', 'L = 400', 'no shifts')
set(gca, 'XScale', 'linear')
set(gca, 'YScale', 'log')
xlabel('\alpha');
ylabel('RMSE (log scale)');
save_name = strcat('XP_EM');
saveas(gcf, strcat(save_name,'.fig'));
saveas(gcf, strcat(save_name,'.jpg'));
pdf_print_code(gcf, strcat(save_name,'.pdf'), 11);


