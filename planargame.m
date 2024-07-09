tic
cvx_clear
py.importlib.import_module('numpy');
n = 1
m = 5
S = "[(1,2), (2,3), (1,4), (1,5)]"
pred = pyrunfile("~/Documents/GitHub/REUGraphGames/planar_test_ml_mod.py", "pred",n = n, m = m, S = S);
prob = pyrunfile("~/Documents/GitHub/REUGraphGames/planar_test_ml_mod.py", "prob",n = n, m = m, S = S);
prob = double(prob);
prob = prob(1, 1);
pred_mat = double(pred);
cvx_begin sdp quiet
    variable p([size(pred_mat, 1), size(pred_mat, 2), size(pred_mat, 3), size(pred_mat, 4)])
    p_value = 0;
    for a = 1:size(pred_mat, 1)
        for b = 1:size(pred_mat, 2)
            for s = 1:size(pred_mat, 3)
                for t = 1:size(pred_mat, 4)
                    if pred_mat(a, b, s, t) == 1
                        p_value = p_value + p(a, b, s, t);
                    end
                end
            end
        end
    end

    maximize prob * p_value

    subject to 
        NPAHierarchy(p, 1) == 1;
cvx_end
cvx_optval
toc