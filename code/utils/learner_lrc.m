function model = learner_lrc(errors_validate, model)
    cv_results = reshape([errors_validate{:, 1}], [numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)]);
    cv_results(numel(model.can_tau_S), :, :) = Inf;
    cv_results(:, :, 1 : numel(model.can_tau_I) - 1) = Inf;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, ~] = ind2sub([numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)], loc_best);

    model.tau_I = 0;
    model.tau_A = model.can_tau_A(d2);
    model.tau_S = model.can_tau_S(d1);
end