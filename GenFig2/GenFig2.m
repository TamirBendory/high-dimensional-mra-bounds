clear;
close all;
clc;

% seeds the random number generator
seed = rng(1234);

%% Generate N observations with noise variance sigma^2

L_vec = [40, 160, 640]; %different signal length
length_L = length(L_vec);
length_alpha = 30;
alpha_vec = linspace(0.5, 5, length_alpha);
num_iter = 50; % number of iterations per alpha
err_geini = zeros(length_L, length_alpha, num_iter);
a = 100; % a scaling parameter for the number of measurements
for l = 1:length_L
    L = L_vec(l); %signal length
    x = randn(L, 1);
    hx = fft(x); % fft of signal
    N = round(a*L/log(L));  %number of measurements
    for n = 1:length_alpha
        alpha = alpha_vec(n);
        fprintf('L = %g, alpha = %.2g\n', L, alpha); 
        sigma = sqrt(L/alpha/log(L)); %noise level
        for iter = 1:num_iter
            % generating data (shifted and noisy measurements)
            shifts = randi([0,L-1],N, 1);
            X = zeros(L, N);
            for i = 1:N
                X(:,i) = circshift(x, shifts(i));
            end
            X = X + sigma*randn(L, N); % noisy data
            hX = fft(X); % fft of the data
            shift_est = zeros(N,1);
            % estimating the signal
            x_est = 0;
            for i = 1:N
                [~, shift_est(i)] =  max(ifft(conj(hx).*hX(:,i))); % estimating shifts
                x_est = x_est + circshift(X(:,i), -shift_est(i)); % aligning measurements
            end
            x_est = x_est/N; %estimated signal
            x_est = align_to_reference(x_est, x);
            err_geini(l, n, iter) = norm(x - x_est)/norm(x);
        end
        clear X
        save('XP_geini.mat');
    end
end

%% plotting

err = mean(err_geini, 3); %average error
ln = 1.3; 
figure;
hold on;
plot(alpha_vec, err(1,:),'linewidth', ln);
plot(alpha_vec, err(2,:),'linewidth', ln);
plot(alpha_vec, err(3,:),'linewidth', ln);
plot(alpha_vec, 1./sqrt(1+ a*alpha_vec),'linewidth', ln);
xline(2,'--r');
xlabel('\alpha');
ylabel('RMSE (log scale)');
set(gca, 'YScale', 'log')
legend('L = 40', 'L = 160', 'L = 640', 'no shifts');
save_name = strcat('XP_genie');
saveas(gcf, strcat(save_name,'.fig'));
saveas(gcf, strcat(save_name,'.jpg'));
pdf_print_code(gcf, strcat(save_name,'.pdf'), 11)
