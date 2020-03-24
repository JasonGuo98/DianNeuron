import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab

__all__ = ["get_graph_noed_and_edge","plot_network"]

color_dic = {"Input":1,"Dense":0.75,"ADD":0.5,"Dropout":0.25}
def get_graph_noed_and_edge(input_layer,output_layer):
    edges = []
    edge_labels = {}
    all_layer = [input_layer]
    for layer in all_layer:
        if hasattr(layer,"activation"):
            layer_name = layer.name + "=>"+layer.activation.name
        else:
            layer_name = layer.name
        if hasattr(layer,"bn_layer"):
                layer_name = layer_name + "=>"+"BatchNormal"
        for next_layer in layer.next_layer_list:
            if hasattr(next_layer,"activation"):
                next_layer_name = next_layer.name + "=>"+next_layer.activation.name
            else:
                next_layer_name = next_layer.name
            if hasattr(next_layer,"bn_layer"):
                next_layer_name = next_layer_name + "=>"+"BatchNormal"
            all_layer.append(next_layer)
            edges.append([layer_name,next_layer_name])
            edge_labels[(layer_name,next_layer_name)] = "BatchSize"+"×"+str(layer.out_dim)
    return edges,edge_labels

def plot_network(input_layer,output_layer):
    G = nx.DiGraph()
    edges,edge_labels = get_graph_noed_and_edge(input_layer,output_layer)
    G.add_edges_from(edges, weight=1)
    
    node_color = []
    for node in G.nodes:
        # print(node)
        got_color = False
        for k,v in color_dic.items():
            if node.startswith(k):
                node_color.append(v)
                got_color = True
                break
        if not got_color:
            node_color.append(0)
    
    pos=nx.spectral_layout(G)
    pylab.figure(figsize=(15,7.5))
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nx.draw(G,pos, with_labels=True,cmap = "rainbow",node_color = node_color, arrowsize = 20,node_size=3000,edge_cmap=plt.cm.Reds)
    pylab.show()