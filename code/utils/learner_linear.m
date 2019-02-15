function model = learner_linear(errors_validate, model)
    cv_results = reshape(errors_validate, [numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)]);
    [~, loc_best] = min(cv_results(numel(model.can_tau_S), :, numel(model.can_tau_I)));

    model.tau_I = 0;
    model.tau_A = model.can_tau_A(loc_best);
    model.tau_S = 0;
end