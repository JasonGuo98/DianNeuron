from .ops_class import *

def add(name='ADD'):
    def inner_build(layers_list):
        return ADD(layers_list,name)
    return inner_build
