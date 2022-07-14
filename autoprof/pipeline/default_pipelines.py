

default_fitting_pipeline = {
    "structure": ["Load_Images", "Variance_Image", "Create_Models_Spec", "Initialize_Models", "fit_loop", "Plot_Loss_History", "Save_Models", "Plot_Model", "Plot_Galaxy_Profiles"],
    "node_kwargs": {
        "fit_loop": {
            "structure": ["Update_Parameters_Random_Grad", "Sample_Models", "Loss_Image", "Compute_Loss", "Lock_Models", ("Stop_Iteration", ("Start", "End"))],
            "node_class": "Chart",
        }
    }
}
    
