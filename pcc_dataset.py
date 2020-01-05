import os
import multiprocessing as mp
from multiprocessing import Pool
import time
import pickle, re

import pandas as pd

from sklearn.model_selection import train_test_split 

from nltk import word_tokenize 

import torch
import torch.utils.data as data


raw_data_path = os.path.join('.', 'Dataset', 'task2_trainset.csv')
testdata_path = os.path.join(".", "Dataset", "task2_public_testset.csv")

splitted_trainingset_path = os.path.join('.', 'Dataset', 'TrainingSet.csv')
splitted_validationset_path = os.path.join('.', 'Dataset', 'ValidationSet.csv')
finetest_path = os.path.join(".", "Dataset", "TestingSet.csv")

fine_subset_path = splitted_trainingset_path, splitted_validationset_path, finetest_path 

num_workers=os.cpu_count()

def read_config():
    config_path=os.path.join('.', 'Configuration', 'Data_Stat.config')
    with open(config_path, 'r') as f:
        lines=f.readlines()
    status=dict()
    
    for line in lines:
        line=line.strip().replace(" ",'')
        v, c=line.split('=')
        status.update({v:eval(c)})
    
    return status
def _init_rawdata(multiworking=False):

    global raw_data_path, testdata_path, splitted_trainingset_path, splitted_validationset_path, finetest_path, fine_subset_path

    status=read_config()
    if status["initialized"]:
        msg="Dataset has been initialized."
        print( msg)
        return msg


    msg="not finish"
    raw_data=pd.read_csv(raw_data_path, dtype=str)
    raw_data.drop(["Title", "Authors","Categories","Created Date"] , axis=1, inplace=True)

    raw_data["THEORETICAL"]=0.0
    raw_data["ENGINEERING"]=0.0
    raw_data["EMPIRICAL"]=0.0
    raw_data["OTHERS"]=0.0

    for i, row in enumerate(raw_data.iterrows()):
        for c in row[1]["Task 2"].split(' '):
            raw_data.loc[i,c]=1

    subset = TrainingSet, ValidationSet = train_test_split(raw_data, test_size=0.2, random_state = 42)


    testdata=pd.read_csv(testdata_path, dtype=str)
    testdata.drop(["Title", "Authors","Categories","Created Date"] , axis=1, inplace=True)
    

    subset.append(testdata)

    

    start_time=time.time()
    #Use multiple processes to produce csv file rather than serial way 
    if multiworking:
        '''
        with Pool(6) as pool:
            
            #for i, df in enumerate(subset):
             #   pool.apply(to_csv, (df, fine_subset_path[i]) )
              
            pool.starmap_async(_to_csv, zip(subset,fine_subset_path))

            pool.close()
            pool.join()'''
        Processes=[mp.Process(target=_to_csv, args=args_set) for args_set in zip(subset, fine_subset_path) ]

        for p in Processes:
            p.start()
        for p in Processes:
            p.join()
        '''    
        p1=mp.Process(target=_to_csv, args=(TrainingSet, splitted_trainingset_path))
        p2=mp.Process(target=_to_csv, args=(ValidationSet, splitted_validationset_path))
        p3=mp.Process(target=_to_csv, args=(testdata, finetest_path ))
        
        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()'''


    else:
        _to_csv(TrainingSet, splitted_trainingset_path )
        _to_csv(ValidationSet, splitted_validationset_path )
        _to_csv(testdata, finetest_path )
        #TrainingSet.to_csv( splitted_trainingset_path )
        #ValidationSet.to_csv( splitted_validationset_path )
        #testdata.to_csv(os.path.join(".", "Dataset", "TestingSet.csv"))
 
    end_time=time.time()

    msg="finsh"
    #print("multiprocessing",multiworking, "time consuming:", end_time-start_time)
    return msg

def _to_csv(df, dest):    
    df.to_csv(dest)
    #print("checkpoint :　_to_csv")


def gathering_words():

    global splitted_trainingset_path, num_workers
    

    traindata = pd.read_csv(splitted_trainingset_path, dtype=str)

    sent_list=list()
    for i, row in enumerate(traindata.iterrows()):
        sent_list.extend(row[1]['Abstract'].split('$$$'))
                                
    word_chunks=[' '.join(sent_list[s:s+(len(sent_list)//num_workers)]) for s in range(0, len(sent_list), len(sent_list)//num_workers )]
  
    with Pool(num_workers) as pool:
        results=pool.map_async(word_tokenize, word_chunks)
        words_collection = sum(results.get(),[])
        #results.get() = [ ['word','...','...' ],[ '...','...','...' ],[ '...','...','...' ],[ '...','...','...' ],[ '...','...','...' ],[ '...','...','...' ],[ '...','...','...' ] ]
        #The first pair of brackets of results.get() presents its iterable property.
        #The element-wise brackets show these 7(means 6 cores) elements' are lists basically. 
        #It leads to the last [] argument is needed because sum assumes it to be 0, and we can't add lists and integers.
    words_collection=set(words_collection)
    
    with open(os.path.join('.', 'Embedding','words_collection.pickle'), 'wb') as fp:
        pickle.dump(words_collection, fp)
    
    return words_collection





class Embedding:#It takes 4.435582876205444 s
    def __init__(self, embedding_path, oov_as_unk=True, lower=True, rand_seed=524):
        
        self.word_dict = {}        
        self.vectors = None
        self.lower = lower 

        self.words=self.load_words_collection()      
        self.extend(embedding_path, self.words, oov_as_unk)

        torch.manual_seed(rand_seed)

        if '<pad>' not in self.word_dict:
            self.add(
                '<pad>', torch.zeros(self.get_dim())
            )
        
        if '<bos>' not in self.word_dict:
            t_tensor = torch.rand((1, self.get_dim()), dtype=torch.float)
            torch.nn.init.orthogonal_(t_tensor)
            self.add( 
                '<bos>', t_tensor
            )
            
        if '<eos>' not in self.word_dict:
            t_tensor = torch.rand((1, self.get_dim()), dtype=torch.float)
            torch.nn.init.orthogonal_(t_tensor)
            self.add(
                '<eos>', t_tensor
            )
        
        if '<unk>' not in self.word_dict:
            self.add('<unk>')

    def load_words_collection(self):
        
        collection_path=os.path.join('.', 'Embedding', 'words_collection.pickle')

        
        with open(collection_path, 'rb') as fc:
            words_collection = pickle.load(fc)


        return words_collection


    def to_index(self, word):
        """
        Args:
            word (str)

        Return:
             index of the word. If the word is not in `words` and not in the
             embedding file, then index of `<unk>` will be returned.
        """
        if self.lower:
            word = word.lower()

        if word not in self.word_dict:
            return self.word_dict['<unk>']
        else:
            return self.word_dict[word]

    def get_dim(self):
        return self.vectors.shape[1]

    def get_vocabulary_size(self):
        return self.vectors.shape[0]

    def add(self, word, vector=None):
        if self.lower:
            word = word.lower()

        if vector is not None:
            vector = vector.view(1, -1)
        else:
            vector = torch.empty(1, self.get_dim())
            torch.nn.init.uniform_(vector)
        self.vectors = torch.cat([self.vectors, vector], 0)
        self.word_dict[word] = len(self.word_dict)

    def extend(self, embedding_path, words, oov_as_unk=True):

        self._load_embedding(embedding_path, words)
        
        if words is not None and not oov_as_unk:
            # initialize word vector for OOV
            for word in words:
                if self.lower:
                    word = word.lower()
                    
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)

            oov_vectors = torch.nn.init.uniform_(
                torch.empty(len(self.word_dict) - self.vectors.shape[0],
                            self.vectors.shape[1]))

            self.vectors = torch.cat([self.vectors, oov_vectors], 0)

    def _load_embedding(self, embedding_path, words):
        
        if words is not None:
            words = set(words)

        vectors = []

        with open(embedding_path) as fp:

            row1 = fp.readline()
            # if the first row is not header
            if not re.match('^[0-9]+ [0-9]+$', row1):
                # seek to 0
                fp.seek(0)
            # otherwise ignore the header

            for i, line in enumerate(fp):
                cols = line.rstrip().split(' ')
                word = cols[0]
                
                if words is not None and word not in words:
                    continue
                elif word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                    vectors.append( list(map(float,cols[1:])) )
                    
        vectors = torch.tensor(vectors)
        if self.vectors is not None:
            self.vectors = torch.cat([self.vectors, vectors], dim=0)
        else:
            self.vectors = vectors


    def __repr__(self):
        return self.vectors

    def __str__(self):
        return str(self.vectors)




embedding_path=os.path.join(".", "Embedding", "glove",'glove.6B.300d.txt')#https://www.kaggle.com/thanakomsn/glove6b300dtxt#glove.6B.300d.txt
Embedder=Embedding(embedding_path)



def get_dataset(datapath):
    global num_workers
    df=pd.read_csv(datapath, dtype=str)
    
    #print("cp")
    chunks=[None for i in range(num_workers+1)]
    #embedders=[embedder for i in range(num_workers+1)]
    for i, batch_start in enumerate(range(0, len(df), len(df)//num_workers)):
        batch_end = batch_start+(len(df)//num_workers) if batch_start+(len(df)//num_workers)<len(df) else len(df)              
        chunks[i]=df[batch_start:batch_end] 
        
    with Pool(num_workers) as pool:
        
        output = pool.map_async(preprocess_samples, chunks )
        result=sum(output.get(),[])
    
    return result





def preprocess_samples(dataset):
    global Embedder

    processed = []
    for row in dataset.iterrows():        
        processed.append(preprocess_sample(row[1], Embedder))
       
   
    return processed

def preprocess_sample(row, embedder):

    result = dict()
    result['Abstract'] = [sentence_to_indices(sentence, embedder) for sentence in row['Abstract'].split('$$$')]    
    if 'Task 2' in row:#in case of handling Testing Data
        result['Label'] = to_onehot(row['Task 2'])


    return result

def sentence_to_indices(sentence, embedder):
    return [embedder.to_index(word) for word in word_tokenize(sentence)]
    
def to_onehot(labels):

    onehot_pos = {'THEORETICAL': 0, 'ENGINEERING':1, 'EMPIRICAL':2, 'OTHERS':3}
    result=[1 if pos in [ onehot_pos[label] for label in labels.split(' ')] else 0 for pos in range(4) ]

    return result





'''
https://zhuanlan.zhihu.com/p/30934236
反複調用DataLoaderIter 的__next__()來得到batch，具體操作就是, 多次調用dataset的__getitem __（）方法
（如果num_worker> 0就多線程調用），然後使用collate_fn來把它們打包成批。中間將會涉及到shuffle，以及sample的方法等
'''
class TextData(data.Dataset):
    def __init__(self, data, max_sentlen=200):
        self.data = data
        self.PAD_TOKEN=1
        self.max_sentlen=max_sentlen


    def __len__(self):
        return len(self.data) 

    def __getitem__(self,index):
        return self.data[index]

    def collate_fn(self, datas):
        
        max_sent = max([len(page['Abstract']) for page in datas])
        max_len = max([min(len(sentence), self.max_sentlen ) for data in datas for sentence in data['Abstract']])
        batch_abstract = []
        batch_label = []
        sent_len = []
        for data in datas:
            # padding abstract to make them in same length
            pad_abstract = []
            for sentence in data['Abstract']:
                if len(sentence) > max_len:
                    pad_abstract.append(sentence[:max_len])
                else:
                    pad_abstract.append(sentence+[self.PAD_TOKEN]*(max_len-len(sentence)))
            sent_len.append(len(pad_abstract))
            pad_abstract.extend([[self.PAD_TOKEN]*max_len]*(max_sent-len(pad_abstract)))
            batch_abstract.append(pad_abstract)
            
            # gather labels
            if 'Label' in data:
                batch_label.append(data['Label'])
                
        return torch.LongTensor(batch_abstract), torch.FloatTensor(batch_label)#, sent_len


