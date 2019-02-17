function models = parameters(data_name)
    if(strcmp(data_name,'dna'))        
        model_lrc_ssl.can_tau_I = 2^-9;
        model_lrc_ssl.can_tau_A = 2^-4;
        model_lrc_ssl.can_tau_S = 2^-9;
        model_lrc_ssl.can_step = 2^4;

        model_lrc.can_tau_I = 2^-9;
        model_lrc.can_tau_A = 2^-4;
        model_lrc.can_tau_S = 2^-9;
        model_lrc.can_step = 2^4;

        model_ssl.can_tau_I = 2^-9;
        model_ssl.can_tau_A = 2^-4;
        model_ssl.can_tau_S = 2^-9;
        model_ssl.can_step = 2^4;

        model_linear.can_tau_I = 2^-9;
        model_linear.can_tau_A = 2^-4;
        model_linear.can_tau_S = 2^-9;
        model_linear.can_step = 2^4;
        models = struct('model_lrc_ssl', model_lrc_ssl, ...
            'model_lrc', model_lrc, ...
            'model_ssl', model_ssl, ...
            'model_linear', model_linear);
    end
end