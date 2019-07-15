import networkx as nx
import pickle, sys, os



def read_data(filename, path):
    """Extract the file and convert it into a neighbor list"""

    g = nx.Graph(name=filename)
    count = 0
    for line in open(path+'\\'+filename):
        #(s, d, _) = line.split(' ')
        count += 1
        try:
            (s, d) = line.split(' ')
        except ValueError:
            print("aa")
        #(s,d) = line.split('\t')
        src = s
        dst = d.split('\n')[0]
        g.add_edge(src, dst)

    pickle.dump(g, open(filename+'nx_graph.pickle','+wb'))


if __name__ == "__main__":

    path = "."
    files = os.listdir(path)
    for name in files:
        if name.endswith(".txt"):
            read_data(name, path)
