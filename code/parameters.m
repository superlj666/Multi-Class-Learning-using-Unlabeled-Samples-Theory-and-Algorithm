function models = parameters(data_name)
    if(strcmp(data_name,'dna'))        
        model_lrc_ssl.can_tau_I = 2^-9;
        model_lrc_ssl..can_tau_A = 2^-4;
        model_lrc_ssl..can_tau_S = 2^-9;
        model_lrc_ssl..can_step = 2^4;
        models = struct{}
    end
end