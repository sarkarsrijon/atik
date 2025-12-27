function [lambda_final, lambda_all, RRE] = atik_nm(A,b,x_true,maxIter,noiseLevel,eta,guess)

breakout = eta*noiseLevel; % "breakout" according to DP
normB = norm(b,2);

rrnorm = zeros(maxIter,1); %preallocate relative res. vec.
Q = zeros(size(b,1),maxIter+1);
Q(:,1) = b/norm(b,2); %assign first column of Q matrix
H = zeros(maxIter,maxIter+1); %preallocate Hessenberg matrix

for i = 1:maxIter
    v = A*Q(:,i);
    for j = 1:i
        H(j,i) = Q(:,j)'*v;
        v = v - H(j,i)*Q(:,j);
    end
    
    H(i+1,i) = norm(v,2);
    Q(:,i+1) = v/H(i+1,i);

    [U, S, ~] = svd(H(1:i+1, 1:i));
    sigma = diag(S);
    k = length(sigma);

    lambda0 = guess;
    bdot = normB*eye(i+1,1);
    bhat = U'*bdot;

    for ell = 1:k-1
        phi = @(lambda0) (bhat(ell)/lambda0 * sigma(ell)^2 + 1)^2 + bhat(k)^2 - breakout^2;
    end

    for ell = 1:k-1
        phi_diff = @(lambda0) ((bhat(ell)^2 * -2 * sigma(ell)^2)/(lambda0 * sigma(ell)^2 + 1)^3);
    end

    [lambda_final, lambda_full] = nm(phi,phi_diff,guess,1e-6,10);
    lambda_all{i} = lambda_full;

    y = (H(1:i+1,1:i)' * H(1:i+1,1:i) + lambda_final * eye(i)) \ (H(1:i+1,1:i)' * bdot);
    xn = Q(1:i)*y;
    RRE(i) = norm(x_true - xn)/norm(x_true);
    
    % Tracking residual
    rrnorm(i) = norm(A*xn - b)/normB;
    
    if rrnorm(i) < breakout
        fprintf('**GMRES terminated at iteration %d** \n',i);
        rrnorm = rrnorm(1:i);
        x = Q(:,1:i)*y; %approximate soln
        iter = i;
        break
    end
    if i == maxIter
        fprintf('**GMRES failed to terminate after maxIters')
        x = Q(:,1:i)*y; %approximate soln
        iter = i;
        break
    end
end