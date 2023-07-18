clc;
clear all;
close all;
 [filename, pathname] = uigetfile({'*.*';'*.png';'*.jpg';'*.tif'}, 'Pick a Image File');
            img = imread([pathname,filename]);
            sig_orig = rgb2gray(img);
sig_orig=double(sig_orig);
[M,N]=size(sig_orig);
sigma = 25;               
win_leng = 4;                                         

h = 2.08*sigma;           

boun_size = 7;               
V = 2;                   
                        
    im_data = sig_orig + sigma*randn(M,N); 

 
im_data = region_outlet(im_data,win_leng,win_leng);
N=N+2*win_leng;
M=M+2*win_leng;
Bs=(2*win_leng+1)^2;
%%%%%%%%%% 
 weighting_function = @(r) (r<h).*(1-(r/h).^2).^8; 

    data_pre_process=im_data; 

    sig_est = zeros(size(im_data,1),size(im_data,2),(2*V+1)^2);

remodify_daq = zeros(size(im_data));
remodify_daq_seq = zeros(size(im_data));      


for row_dat = -boun_size:boun_size,
    for col_dat = -boun_size:boun_size,          
        if row_dat>0 || (row_dat==0 && col_dat>0),
            m_min = max(min(win_leng-row_dat,M-win_leng),win_leng+1);
            m_max = min(max(M-win_leng-1-row_dat, win_leng),M-win_leng-1);
            n_min = max(min(win_leng-col_dat, N-win_leng),win_leng+1);
            n_max = min(max(N-win_leng-1-col_dat, win_leng),N-win_leng-1);
            if n_min>n_max || m_min>m_max,
                continue;
            end;             
                row_length = 1+(m_min:m_max);
                colum_length = 1+(n_min:n_max);
                ssd = zeros(M,N);
                ssd(row_length,colum_length) = conv2 (conv2 ((data_pre_process(row_length,colum_length) - data_pre_process(row_length+row_dat,colum_length+col_dat)).^2, ...
                        ones(1, 2*win_leng+1), 'same'), ones(2*win_leng+1, 1), 'same');                
            row_length = 1+(m_min:m_max);
            colum_length = 1+(n_min:n_max);
            weights = weighting_function(sqrt(ssd(row_length,colum_length)/Bs));
                remodify_daq(row_length,colum_length)       = remodify_daq(row_length,colum_length) + weights;
                remodify_daq(row_length+row_dat,colum_length+col_dat) = remodify_daq(row_length+row_dat,colum_length+col_dat) + weights;
                remodify_daq_seq(row_length,colum_length)    = remodify_daq_seq(row_length,colum_length) + weights.^2;
                remodify_daq_seq(row_length+row_dat,colum_length+col_dat) = remodify_daq_seq(row_length+row_dat,colum_length+col_dat) + weights.^2;

                for m=-V:V,
                    for n=-V:V
                        index=(V+m)*(2*V+1)+V+n+1;
                        sig_shifted = circshift(im_data,[m n]);
                        sig_est(row_length,colum_length,index)=sig_est(row_length,colum_length,index)+weights.*sig_shifted(row_length+row_dat,colum_length+col_dat);
                        sig_est(row_length+row_dat,colum_length+col_dat,index)=sig_est(row_length+row_dat,colum_length+col_dat,index)+weights.*sig_shifted(row_length,colum_length);
                    end;
                end;
           
        elseif row_dat==0 && col_dat == 0,
            weight = 0.01*weighting_function(0);
            remodify_daq = remodify_daq + weight*ones(size(remodify_daq));
            remodify_daq_seq = remodify_daq_seq + weight^2*ones(size(remodify_daq));
                for m=-V:V,
                    for n=-V:V
                        index=(V+m)*(2*V+1)+V+n+1;
                        sig_est(:,:,index) = sig_est(:,:,index) + weight*circshift(im_data,[m n]);
                    end;
                end;
            end;
       
    end;
end;
    B = 16; 
    local_noise_var = sigma.^2 * remodify_daq_seq./max(1e-10,remodify_daq).^2;
    for m=1:B:size(sig_est,1)
        for n=1:B:size(sig_est,2)
            block_rm = m:min(m+B-1,M);
            block_rn = n:min(n+B-1,N);
            sample = sig_est(block_rm, block_rn,:);
            for k=1:size(sample,3)
                sample(:,:,k) = sample(:,:,k) ./ max(1e-10,remodify_daq(block_rm, block_rn));
            end;
            sample = reshape(sample,[size(sample,1)*size(sample,2), size(sample,3)]);
            y_mean = kron(mean(sample,2),ones(1,size(sample,2)));
            Cy = cov(sample - y_mean);
            [U,Sig_Y_diag,dummy] = svd(Cy);
            Sig_Y_diag = diag(Sig_Y_diag).';
            u = (sample - y_mean) * U; 
            for m1=block_rm,
                for n1=block_rn,
                    Sig_X_diag = max(1e-2,Sig_Y_diag-local_noise_var(m1,n1));
                    index=(m1-m)*length(block_rn)+ n1-n+1;
                    u(index,:) = u(index,:) .* Sig_X_diag ./ (Sig_X_diag + local_noise_var(m1,n1));
                end;
            end;
            y = u * U' + y_mean; 
            y = reshape(y,[length(block_rm),length(block_rn),size(sample,2)]);
            
            for k=1:size(y,3)
                y(:,:,k) = y(:,:,k) .* max(1e-10,remodify_daq(block_rm, block_rn));
            end;
           
            sig_est(block_rm,block_rn,:) = y;
        end;
    end;

    remodify_daq = max(1e-20,conv2(remodify_daq,ones(2*V+1,2*V+1),'same'));
    sig_final = zeros(size(im_data));
    for m=-V:V,
        for n=-V:V
            index=(V+m)*(2*V+1)+V+n+1;
            sig_final = sig_final + circshift(sig_est(:,:,index),-[m n]) ./ remodify_daq;
        end;
    end;
    sig_est = sig_final;
    clear sig_final;
sig_est = sig_est(win_leng+1:end-win_leng,win_leng+1:end-win_leng);
im_data = im_data(win_leng+1:end-win_leng,win_leng+1:end-win_leng);
figure('name','Speckle Noise Analysis');
subplot(311),imshow(sig_orig,[]),title(sprintf('Input image'));
subplot(312),imshow(im_data,[]),title(sprintf('Noisy image PSNR=%f dB', psn_img(im_data,sig_orig)));
subplot(313),imshow(sig_est,[]),title(sprintf('Denoised image PSNR=%f dB', psn_img(sig_est,sig_orig)));
[ind,maap]=sim_ind(im_data,sig_orig);
fprintf('Noisy image SSIM ind=%f %',ind*100);
fprintf('\n\n');
[ind,maap]=sim_ind(sig_est,sig_orig);

fprintf('Denoisy image SSIM ind=%f %',ind*100);
fprintf('\n\n');

