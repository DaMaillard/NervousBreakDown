

M = imread('train/1_1.tif');

size_dim1 = 100;
size_dim2 = 100;
overlap_dim1 = .5;
overlap_dim2 = .5;
[featM, nbim_dim1, nbim_dim2] = feat_extract_image(M, size_dim1, size_dim2, ...
                                                    overlap_dim1, overlap_dim2);

figure(1)
imagesc(M);

figure(2)

k=1;

for m=1:nbim_dim1
    for p=1:nbim_dim2
        M_mp = reshape(featM(k,:), size_dim1, size_dim2);
        subplot(nbim_dim1, nbim_dim2, k);
        imagesc(M_mp);

        k = k+1;
    end
end
