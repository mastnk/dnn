from .vgg import VGG

def gen_model( name, **cfg ):
    name = name.lower()
    if( name.startswith('vgg') ):
        model = VGG(name.upper())
    else:
        raise NotImplementedError()

    return model
