function RunFullAssemblyAnalysis(pairsOfContacts, CM)
    % RUNFULLASSEMBLYANALYSIS Performs individual Force Closure analysis 
    % for each body and a final combined assembly visualization.
    %
    % pairsOfContacts: [n x 6] -> [body1, body2, x, y, alpha, mu]
    % CM: [m x 3] -> [x_cm, y_cm, mass]

    num_bodies = size(CM, 1);
    
    % 1. Generate Individual Force Closure Plots for each body
    for b_id = 1:num_bodies
        VisualizeBodyForceClosure(pairsOfContacts, CM, b_id);
    end
    
    % 2. Generate Combined Global Assembly View
    VisualizeGlobalAssembly(pairsOfContacts, CM);
end