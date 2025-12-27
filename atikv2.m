function [alpha] = atikv2(alpha0, bhat, S, eta, noiseLevel)

    syms alpha_sym;
    
    D = S' * S; len = size(D, 1);
    Phi = S * inv(D + alpha_sym * eye(len)) * S';
    Phi_I_sym = eye(size(Phi)) - Phi;

    f_sym = ((Phi_I_sym * bhat)' * (Phi_I_sym * bhat)) - (eta * noiseLevel)^2;

    % compute the derivative of f(alpha)
    df_sym = diff(f_sym, alpha_sym);

    % convert symbolic expressions to MATLAB functions
    f = matlabFunction(f_sym, 'Vars', alpha_sym);
    df = matlabFunction(df_sym, 'Vars', alpha_sym);
    
    % root finding using Newton's method
    [alpha] = nm(f, df, alpha0);

    if alpha >= 10e15
        alpha = 0;
    else
        alpha = 1/alpha;
    end
    

end
