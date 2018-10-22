import pickle
import numpy
class API:
	def __init__(self,model_filename,id_limit=1000,date_base=2018,date_length=360):
		self.model_filename = model_filename
		self.knn_clf = None
		self.id_limit = id_limit
		self.date_base = date_base
		self.date_length = date_length
		
		with open(self.model_filename,'rb')as f:
			self.knn_clf = pickle.load(f)
			
	def normal_id(self,_id,limit):
		return int(_id)/limit
	
	def normal_date(self,date,base,length):
		n_time = np.sum([int(num)for num in date.split('-')]*np.array([360,30,1]))
		return (n_time-base*360)/length
	
	def predict(self,x):
		f1 = self.normal_id(x[0],self.id_limit)
		f2 = self.normal_date(x[1],self.date_base,self.date_length)
		X = [[f1,f2]]
		return self.knn_clf.predict(X).item()
if __name__ == '__main__':
	api = API('../knn_clf.pkl')
	_id = 100
	date = '2018-10-10'
	api.predict([_id,date])
