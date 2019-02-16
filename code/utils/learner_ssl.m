function model = learner_ssl(errors_validate, model)
    cv_results = reshape([errors_validate{:, 1}], [numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)]);
    cv_results(1 : numel(model.can_tau_S) - 1, :, :) = Inf;
    cv_results(:, :, numel(model.can_tau_I)) = Inf;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3] = ind2sub([numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)], loc_best);

    model.tau_I = model.can_tau_I(d3);
    model.tau_A = model.can_tau_A(d2);
    model.tau_S = 0;
end