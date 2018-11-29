import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

if __name__ == '__main__':
    dict = unpickle('/Users/FatemaK/Desktop/year3/SSA/computer-vision/Assignment/pedestrian/INRIAPerson/cifar-100-python/train')
    dict2 = unpickle('/Users/FatemaK/Desktop/year3/SSA/computer-vision/Assignment/pedestrian/INRIAPerson/cifar-100-python/meta')
