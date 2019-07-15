# -*- coding: utf-8 -*-


import Test_set  # The method of import local Python file


# Define the parameter class
class Parakeyward:
    """ Parameter Class """
    def __init__(self, path, sample_p, p, q, num_walks, walks_length, r, examnum, dimensions, workers):
        self.path = path  # The file of path
        self.sample_p = sample_p  # the sample proportion for testing alogrithm
        self.p2return = p  # the parameter of return
        self.q2return = q  # the parameter of In-Out
        self.num_walks = num_walks  # the windows size of Skip-gram
        self.walks_length = walks_length  # the length of walk, like as the sequence of word
        self.radio = r  # the jump propotation of interlayers
        self.examnum = examnum
        self.dimensions = dimensions
        self.workers = workers


# The main function for code
if __name__ == "__main__":
    file = ["CS-Aarhus", "EUAir", "Pierreauger", "CKM", "ArXiv"]
    # file = ["ArXiv"]# ,"Pierreauger", "CKM", "ArXiv"]
    dimension = [100]
    datasets_len = len(file)
    for i in range(datasets_len):
        for ff in range(5,41,5):
            print("------------------%s------------------"%file[i])
            path = "pickle/" + file[i]
            #参数对应           路径,采样,p, q, nw, wl, rj, exam,dim,ws
            print("-------------%s-------------"%str(ff))
            args = Parakeyward(path, 1, 1, 2, 10, 35, 0.7, 5, dimension[i], 4)
            MN = Test_set.Mergeing_vec_N2V(args.path, 0.7, 0.2, args.sample_p, args.p2return, args.q2return, args.num_walks, args.walks_length, args.radio, args.examnum, args.dimensions, args.workers, 0.25)
            MN.run()
