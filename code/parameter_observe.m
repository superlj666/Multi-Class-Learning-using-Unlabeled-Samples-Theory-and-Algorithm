function parameter_observe(data_name)
    data_name = 'dna';
    load(['../data/', data_name, '/', 'cross_validation.mat']);

    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, numel(can_tau_S), :, :) = Inf;
    cv_results(:, :, :, numel(can_tau_I)) = Inf;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----LRC_SSL: %.4f\t tau_I: %.0f\t tau_A: %.0f\t tau_S: %.0f\t step: %.1f\n-----\n', ...
    errors_validate{loc_best, 1}, log(can_tau_I(d4))/log(2), log(can_tau_A(d3))/log(2), log(can_tau_S(d2))/log(2), log(can_step(d1))/log(2));
    min_lrc_ssl = errors_validate{loc_best, 1};

    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, numel(can_tau_S), :, :) = Inf;
    cv_results(:, :, :, 1 : numel(can_tau_I) - 1) = Inf;
    cv_results(cv_results <= min_lrc_ssl) = Inf;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----LRC: %.4f\t tau_I: %.0f\t tau_A: %.0f\t tau_S: %.0f\t step: %.1f\n-----\n', ...
    errors_validate{loc_best, 1}, log(can_tau_I(d4))/log(2), log(can_tau_A(d3))/log(2), log(can_tau_S(d2))/log(2), log(can_step(d1))/log(2));

    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, 1 : numel(can_tau_S) - 1, :, :) = Inf;
    cv_results(:, :, :, numel(can_tau_I)) = Inf;
    cv_results(cv_results <= min_lrc_ssl) = Inf;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----SSL: %.4f\t tau_I: %.0f\t tau_A: %.0f\t tau_S: %.0f\t step: %.1f\n-----\n', ...
    errors_validate{loc_best, 1}, log(can_tau_I(d4))/log(2), log(can_tau_A(d3))/log(2), log(can_tau_S(d2))/log(2), log(can_step(d1))/log(2));

    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, 1 : numel(can_tau_S) - 1, :, :) = Inf;
    cv_results(:, :, :, numel(can_tau_I)) = Inf;
    cv_results(:, numel(can_tau_S), :, :) = Inf;
    cv_results(:, :, :, 1 : numel(can_tau_I) - 1) = Inf;
    cv_results(cv_results <= min_lrc_ssl) = Inf;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----Linear: %.4f\t tau_I: %.0f\t tau_A: %.0f\t tau_S: %.0f\t step: %.1f\n-----\n', ...
    errors_validate{loc_best, 1}, log(can_tau_I(d4))/log(2), log(can_tau_A(d3))/log(2), log(can_tau_S(d2))/log(2), log(can_step(d1))/log(2));
end