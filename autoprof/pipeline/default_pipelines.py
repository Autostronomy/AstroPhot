

default_fitting_pipeline = {'main': ['load images', 'gaussian psf', "create models", 'initialize models', 'fit loop', 'quality checks', 'save models'],
                            'fit loop': ['sample models', 'project to image', 'select models', 'compute loss', 'random update parameters', 'stop iteration']}

default_forced_pipeline = {'main': ['load images', 'gaussian psf', 'load models', 'fit loop', 'quality checks', 'save models'],
                           'fit loop': ['sample models', 'project to image', 'select models', 'compute loss', 'random update parameters', 'stop iteration']}
    
