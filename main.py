import numpy as np 
import pandas as pd


class Dataset:
    def __init__(self, path):
        assert path is not None
        self.train = pd.read_csv(path + 'train_test.csv')
        self.text = self.train.text
        self.label = self.train.label
        
        self.eval = pd.read_csv(path + 'evaluate.csv')
        self.eval_text = self.eval.text
        
        
        ## Sampling 
        idx = np.random.choice(len(self.text), replace=False, size=int(0.2*len(self.text)))
        self.test_text = self.text.iloc[idx]
        self.test_label = self.label.iloc[idx]
        
        rest_idx = np.arange(0, len(self.text))
        rest_idx = np.delete(rest_idx, idx)
        
        self.train_text = self.text.iloc[rest_idx]
        self.train_label = self.label.iloc[rest_idx]
        
        
        print(len(self.train_label), len(self.train_text))
        print(len(self.test_label), len(self.test_text))
        
        self.dictionary = {}
        self.hafez_dict = {}
        self.saadi_dict = {}
        
        self.extract_words()
        
        # for data in self.text:
        #     print(data.split(" "))
        
    def extract_words(self):
        self.hafez_dict = {}
        self.saadi_dict = {}
        for d, l in zip(self.text, self.label):
            words = d.split(" ")
            if l == "hafez":
                for w in words:
                    try:
                        self.hafez_dict[w] += 1
                    except KeyError:
                        self.hafez_dict[w] = 1
            elif l == 'saadi':
                for w in words:
                    try:
                        self.saadi_dict[w] += 1
                    except KeyError:
                        self.saadi_dict[w] = 1
            
        hafez_c = self.hafez_dict.__len__()
        saadi_c = self.saadi_dict.__len__()   
        m = 0         
        for i in self.hafez_dict:
            self.hafez_dict[i] /= hafez_c
                            
        for i in self.saadi_dict:
            self.saadi_dict[i] /= saadi_c  
            if self.saadi_dict[i] > m:
                m = self.saadi_dict[i]  

        # print(m)
    
class Predictor:
    def __init__(self, dataset):
        self.dataset = dataset
        
        # Prior Knowledge
        self.p_hafez = 0
        self.p_saadi = 0
        
    def priorKnowledge(self):        
        label = list(self.dataset.label)

        self.p_hafez = label.count('hafez') / len(label)
        self.p_saadi = label.count('saadi') / len(label)
        
        # print('P hafez:', self.p_hafez)
        # print('P saadi:', self.p_saadi)
        
    def predict(self, text):
        hafez_dict = self.dataset.hafez_dict
        saadi_dict = self.dataset.saadi_dict
        words = text.split(" ")
        hafez_prob = 0
        saadi_prob = 0
        
        ## Calculate P(Hafez|words)
        for w in words:
            tmp = self.p_hafez
            try:
                tmp *= hafez_dict[w]
            except KeyError:
                # print("Not in saadi database")
                # pass
                tmp *= 0
            hafez_prob += tmp
        
        ## Calculate P(Saadi|words)
        for w in words:
            tmp = self.p_saadi
            try:
                tmp *= saadi_dict[w]
            except KeyError:
                # pass
                tmp *= 0
            saadi_prob += tmp
        
        if saadi_prob > hafez_prob:
            return "saadi"
        else:
            return "hafez"
        
    def accuracy(self):
        pass
        
    
    
    
if __name__ == '__main__':
    dataset = Dataset('./Data/')
    # print(dataset.label)
    
    pred = Predictor(dataset)
    pred.priorKnowledge()

    a = list(map(pred.predict, dataset.text))
    print((a == dataset.label).mean())
    



