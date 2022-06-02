

default_fitting_pipeline = {
    "structure": ["Load_Images", "Create_Models_Spec", "Initialize_Models", "fit loop", "Save_Models", "Plot_Loss_History", "Plot_Model"],
    "node_kwargs": {
        "fit loop": {
            "structure": ["Update_Parameters_Random_Grad", "Sample_Models", "Loss_Image", "Compute_Loss", "Lock_Models", ("Stop_Iteration", ("Start", "End"))],
            "node_class": "Chart",
        }
    }
}
    
