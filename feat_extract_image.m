function [Xfeat, nbim_dim1, nbim_dim2] = feat_extract_image(M, size_dim1, size_dim2, overlap_dim1, overlap_dim2)
%
%
%



[n1, n2] = size(M);

% nb de pixels dans la petite image qui ne sont pas pris dans l'overlap, dans la dimension 1
nol_size_1 = floor(size_dim1 * (1-overlap_dim1));
% nb de petites images dans la dimension 1
nbim_dim1 = floor((n1 - size_dim1) / nol_size_1) +1;

% idem pour la dimension 2
nol_size_2 = floor(size_dim2 * (1-overlap_dim2));
nbim_dim2 = floor((n2 - size_dim2) / nol_size_2) +1;


Xfeat = [];

for m = 1:nbim_dim1
    m1 = (m-1)*nol_size_1 + 1;
    m2 = min(m1 + size_dim1 - 1, n1);

    for p = 1:nbim_dim2
        p1 = (p-1)*nol_size_2 + 1;
        p2 = min(p1 + size_dim2 - 1, n2);

        M_mp = M(m1:m2, p1:p2);

        feat_mp = features_extract(M_mp);
        Xfeat = [Xfeat; feat_mp(:)'];
    end
end



