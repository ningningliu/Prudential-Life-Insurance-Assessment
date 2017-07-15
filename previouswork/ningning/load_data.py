import pandas as pd

# Load data file

class DataLoader:
    def __init__(self, path):
        self.path = path
        # print('in DataLoader')

    def loader(self):
        # print('Loading data.')
        file = pd.read_csv(self.path)
        # print('Finish loading')
        return file


train = DataLoader(path='train.csv')
train_data= train.loader()
test = DataLoader(path='test.csv')
test_data = test.loader()
data = train_data.append(test_data)
# print(data.head)
# print(data.shape)


# factorize categorical variables
data['Product_Info_2'] = pd.factorize(data['Product_Info_2'])[0]
# print(data['Product_Info_2'].unique())

data = data.drop('Id',axis =1)








