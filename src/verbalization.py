import pandas as pd
import numpy as np
import copy


#Class storing information about a chunk
class Chunk:

	def __init__(self,feature,value,x_min,x_max,y_min,y_max,mean,pcc,gl_x_min,gl_x_max,gl_y_min,gl_y_max,global_ymin,global_ymax,cutoff,
		denomination="",val_den="",friendly="",include_pcc=False,detailed=False,include_slope=True,include_mean=True,threshold=False,global_norm=False):
		self.feature=feature
		self.friendly=friendly
		if(friendly==""):
			self.friendly=feature
		self.value=value
		self.denomination=denomination
		self.val_den=val_den
		self.x_min=x_min
		self.x_max=x_max
		self.y_min=y_min
		self.y_max=y_max
		self.mean=mean
		self.gl_x_min=gl_x_min
		self.gl_x_max=gl_x_max
		self.gl_y_min=gl_y_min
		self.gl_y_max=gl_y_max
		self.global_ymin=global_ymin
		self.global_ymax=global_ymax
		self.pcc=pcc
		self.threshold=threshold
		self.global_norm=global_norm
		self.include_pcc=include_pcc
		self.include_slope=include_slope
		self.include_mean=include_mean
		self.cutoff=cutoff
		#thresholding, slope and mean not implemented together
		if(self.threshold and self.include_mean):
			self.include_slope=False
		#should chunks be combined
		self.detailed=detailed
		self.impact=abs(y_max-y_min)
		self.sign=0
		if(y_min!=y_max):
			self.sign=abs(y_max-y_min)/(y_max-y_min)
		self.slope=(self.sign*self.impact/(self.x_max - self.x_min))
		self.normalizer=0
		if(gl_y_max-gl_y_min>0):
			if(self.global_norm):
				self.normalizer=abs((gl_x_max-gl_x_min)/(global_ymax-global_ymin))
			else:
				self.normalizer=abs((gl_x_max-gl_x_min)/(gl_y_max-gl_y_min))

	def isImp(self):
		#If a chuck is important to show in the verbalization
		imp = True
		denom=(self.gl_y_max - self.gl_y_min)
		if(self.global_norm):
			denom=self.global_ymax-self.global_ymin
		if(self.gl_y_max==self.gl_y_min):
			imp = False
		elif((not self.detailed)  and abs(self.y_max - self.y_min)/denom < self.cutoff):
			imp=False
		return imp


	def toText(self,short=True):
		#if the feature is zero always, skip the feature
		if((self.gl_y_max==self.gl_y_min)):
			return ""
		text = ""
		#short determines the start of the sentence
		if(short):
			text+= "From "
		else:
			text+= "As " + self.friendly + " changes from "

		text+= "{:.2f}".format(self.x_min) + self.denomination + " to " + "{:.2f}".format(self.x_max) + self.denomination + " " + self.value
		
		if(self.sign == 0):
			text+= " remains constant"
			if(self.include_mean):
				text+=" at "+self.mean
		
		elif(self.threshold):
			if(self.normalizer*self.slope>1):
				text+=" increases rapidly"
				if(self.include_mean):
					text+= " from " + "{:.2f}".format(self.y_min) + " to " + "{:.2f}".format(self.y_max) + " at mean " + "{:.2f}".format(self.mean)
				if(self.include_slope):
					text+= " and is " + "{:.2f}".format(self.slope) + " times " + "{:.2f}".format(self.x_max - self.x_min)
			
			elif(self.normalizer*self.slope<=1 and self.normalizer*self.slope>0.25):
				text+=" increases moderately"
				if(self.include_mean):
					text+= " from " + "{:.2f}".format(self.y_min) + " to " + "{:.2f}".format(self.y_max) + " at mean " + "{:.2f}".format(self.mean)
				if(self.include_slope):
					text+= " and is " + "{:.2f}".format(self.slope) + " times " + "{:.2f}".format(self.x_max - self.x_min)
			
			elif(self.normalizer*self.slope<=0.25 and self.normalizer*self.slope>=-0.25):
				text+=" remains virtually constant"
				if(self.include_mean):
					text+= " at " + "{:.2f}".format(self.mean)
			
			elif(self.normalizer*self.slope<-0.25 and self.normalizer*self.slope>=-1):
				text+=" decreases moderately"
				if(self.include_mean):
					text+= " from " + "{:.2f}".format(self.y_min) + " to " + "{:.2f}".format(self.y_max) + " at mean " + "{:.2f}".format(self.mean)
				if(self.include_slope):
					text+= " and is " + "{:.2f}".format(self.slope) + " times " + "{:.2f}".format(self.x_max - self.x_min)
			
			else:
				text+=" decreases rapidly"
				if(self.include_mean):
					text+= " from " + "{:.2f}".format(self.y_min) + " to " + "{:.2f}".format(self.y_max) + " at mean " + "{:.2f}".format(self.mean)
				if(self.include_slope):
					text+= " and is " + "{:.2f}".format(self.slope) + " times " + "{:.2f}".format(self.x_max - self.x_min)
		
		elif(self.include_slope):
			text+= " changes with slope " + "{:.2f}".format(self.slope) 
		
		else:
			if(self.sign==1):
				text+= " increases "
			else:
				text+= " decreases "
			text+= "by " + "{:.2f}".format(self.impact) + self.val_den 
			if(self.include_pcc):
				text+=" with a pcc score of " + "{:.2f}".format(self.pcc)
		
		text+= ".\n"
		
		return text

def combineChunks(ch1,ch2):
	x_min=min(ch1.x_min,ch2.x_min)
	x_max=max(ch1.x_max,ch2.x_max)
	if(ch1.x_min<ch2.x_min):
		y_min=ch1.y_min
		y_max=ch2.y_max
	else:
		y_min=ch2.y_min
		y_max=ch1.y_max
	mean=((ch1.x_max-ch1.x_min)*ch1.mean+(ch2.x_max-ch2.x_min)*ch2.mean)/(ch1.x_max-ch1.x_min+ch2.x_max-ch2.x_min)
	pcc=((ch1.x_max-ch1.x_min)*ch1.pcc+(ch2.x_max-ch2.x_min)*ch2.pcc)/(ch1.x_max-ch1.x_min+ch2.x_max-ch2.x_min)
	return Chunk(ch1.feature,ch1.value,x_min,x_max,y_min,y_max,mean,pcc,ch1.gl_x_min,ch1.gl_x_max,ch1.gl_y_min,ch1.gl_y_max,
		ch1.global_ymin,ch1.global_ymax,ch1.cutoff,ch1.denomination,ch1.val_den,ch1.friendly,ch1.include_pcc,ch1.detailed,
		ch1.include_slope,ch1.include_mean,ch1.threshold,ch1.global_norm)


#class storing the information about a feature
class Feature:

	def __init__(self,feature,value,x_min,x_max,y_min,y_max,global_ymin,global_ymax,start_y,cutoff=0.1,
		denomination="",val_den="",friendly="",
		include_pcc=False,detailed=False,include_slope=True,include_mean=True,threshold=False,global_norm=False):
		self.feature=feature
		self.friendly=friendly
		if(friendly==""):
			self.friendly=feature
		self.value=value
		self.denomination=denomination
		self.val_den=val_den
		self.x_min=x_min
		self.x_max=x_max
		self.y_min=y_min
		self.y_max=y_max
		self.start_y=start_y
		self.chunks=[]
		self.global_ymin=global_ymin
		self.global_ymax=global_ymax
		self.threshold=threshold
		self.cutoff=cutoff
		self.global_norm=global_norm
		self.include_pcc=include_pcc
		self.include_slope=include_slope
		self.include_mean=include_mean
		if(self.threshold and self.include_mean):
			self.include_slope=False
		self.detailed=detailed

	def combineChunks(self):
		ans=[]
		pre=False
		for chunk in self.chunks:
			ch = chunk
			if(pre):
				ch = combineChunks(prev,chunk)
			if(ch.isImp()):
				pre =  False
				ans.append(ch)
			else:
				prev=ch
				pre=True
		if(pre):
			ans.append(prev)
		return ans


	def toText(self):
		if(not self.detailed):
			self.chunks = self.combineChunks()
		text = ""
		intro = "At " + "{:.2f}".format(self.x_min) + self.denomination + " the " + self.value + " is " + "{:.2f}".format(self.start_y) + self.val_den + ".\n"
		ran = self.value + " ranges from " + "{:.2f}".format(self.y_min) + self.val_den + " to " + "{:.2f}".format(self.y_max) + self.val_den + ".\n"
		short = True
		for chunk in self.chunks:
			text += chunk.toText(not short)
			if(text!=""):
				short = False
		if(text!=""):
			text=intro+ran+text
		return text

	def addChunk(self,x_min,x_max,y_min,y_max,mean,pcc):
		chunk = Chunk(self.feature,self.value,x_min,x_max,y_min,y_max,mean,pcc,self.x_min,self.x_max,self.y_min,self.y_max,
			self.global_ymin,self.global_ymax,self.cutoff,self.denomination,self.val_den,self.friendly,self.include_pcc,self.detailed,
			self.include_slope,self.include_mean,self.threshold,self.global_norm)
		self.chunks.append(chunk)


alias = {}
alias["rm"] = "# Rooms in House"
alias["age"] = "% Pre-1940 Units"
alias["dis"] = "Distance to Business District" 
alias["rad"] = "Highway Accessibility"
alias["ptratio"] = "Student-Teacher Ratio"
alias["tax"] = "Property Tax Rate" 
alias["zn"] = "% Residential in Area"
alias["indus"] = "% Business in Area"
alias["black"] = "Prevalence of Minorities"
alias["lstat"] = "Percent Lower Income in Area"
alias["crim"] = "Crime Rate"
alias["nox"] = "Air Pollution"
combine = False
df = pd.read_csv("../Boston_Housing/sorted_f_59.0.csv")

min_x_dict = {}
max_x_dict = {}
min_y_dict = {}
max_y_dict = {}
start_y = {}
global_max_y = -float('inf')
global_min_y = float('inf')
for index, row in df.iterrows():
	name=row[0]
	x_min=row[1]
	x_max=row[2]
	y_min=row[3]
	y_max=row[4]
	global_max_y=max(global_max_y,max(y_max,y_min))
	global_min_y=min(global_min_y,min(y_max,y_min))
	if(name in min_x_dict):
		min_x_dict[name]=min(min_x_dict[name],x_min)
		max_x_dict[name]=max(max_x_dict[name],x_max)
		min_y_dict[name]=min(min_y_dict[name],min(y_min,y_max))
		max_y_dict[name]=max(max_y_dict[name],max(y_max,y_min))
	else:
		start_y[name]=y_min
		min_x_dict[name]=x_min
		max_x_dict[name]=x_max
		min_y_dict[name]=min(y_min,y_max)
		max_y_dict[name]=max(y_max,y_min)

fts = []
verbs=[]
for gl in range(2):
	for detail_in in range(2):
		for slope_in in range(2):
			for thresh_in in range(2):
				for inc_mean in range(2):
					global_norm=True
					if(gl==0):global_norm=False
					detailed=True
					if(detail_in==0):detailed=False
					include_slope=True
					if(slope_in==1):include_slope=False
					threshold=True
					if(thresh_in==0):threshold=False
					include_mean=True
					if(inc_mean==0):include_mean=False
					pre=False		
					cur_ft=None
					cur_feature=""
					for index, row in df.iterrows():
						name=row[0]
						x_min=row[1]
						x_max=row[2]
						y_min=row[3]
						y_max=row[4]
						mean=row[5]
						pcc=row[8]
						if(cur_feature!=name):
							if(cur_ft):
								fts.append(cur_ft)
							cur_ft = Feature(name,"price",min_x_dict[name],max_x_dict[name],min_y_dict[name],max_y_dict[name]
								,global_min_y,global_max_y,start_y[name],friendly=alias[name],threshold=threshold,
								include_slope=include_slope,include_mean=include_mean,detailed=detailed,global_norm=global_norm)
							cur_feature=name
						cur_ft.addChunk(x_min,x_max,y_min,y_max,mean,pcc)
					fts.append(cur_ft)

for ft in fts:
	if(ft.toText()==""):
		continue
	
	ft.chunks = ft.combineChunks()
	intro = "At " + "{:.2f}".format(ft.x_min) + ft.denomination + " the " + ft.value + " is " + "{:.2f}".format(ft.start_y) + ft.val_den + ".\n"
	ran = ft.value + " ranges from " + "{:.2f}".format(ft.y_min) + ft.val_den + " to " + "{:.2f}".format(ft.y_max) + ft.val_den + ".\n"
	obj={}
	obj["Feature Name"]=ft.feature
	obj["Text"]=intro
	obj["Impact"]=0
	obj["Slope"]=0
	obj["pcc"]=0
	obj["x_min"]=-1
	obj["x_max"]=-1
	obj["Detailed"]="None"
	if(ft.detailed):obj["Detailed"]="Detailed"
	obj["Slope/Impact"]="Impact"
	if(ft.include_slope):obj["Slope/Impact"]="Slope"
	obj["Threshold"]="None"
	if(ft.threshold):obj["Threshold"]="Threshold"
	obj["Include Mean"]="No"
	if(ft.include_mean):obj["Include Mean"]="Yes"
	obj["ShareY"]="Local"
	if(ft.global_norm):obj["ShareY"]="Global"
	verbs.append(obj)
	obj={}
	obj["Feature Name"]=ft.feature
	obj["Text"]=ran
	obj["Impact"]=0
	obj["Slope"]=0
	obj["pcc"]=0
	obj["x_min"]=-1
	obj["x_max"]=-1
	obj["Detailed"]="None"
	if(ft.detailed):obj["Detailed"]="Detailed"
	obj["Slope/Impact"]="Impact"
	if(ft.include_slope):obj["Slope/Impact"]="Slope"
	obj["Threshold"]="None"
	if(ft.threshold):obj["Threshold"]="Threshold"
	obj["Include Mean"]="No"
	if(ft.include_mean):obj["Include Mean"]="Yes"
	obj["ShareY"]="Local"
	if(ft.global_norm):obj["ShareY"]="Global"
	verbs.append(obj)
	text = ""
	short = True
	for chunk in ft.chunks:
		text = chunk.toText(not short)
		if(text!=""):
			obj={}
			obj["Feature Name"]=ft.feature
			obj["Text"]=text
			obj["Impact"]=chunk.impact
			obj["Slope"]=chunk.slope
			obj["pcc"]=chunk.pcc
			obj["x_min"]=chunk.x_min
			obj["x_max"]=chunk.x_max
			obj["Detailed"]="None"
			if(ft.detailed):obj["Detailed"]="Detailed"
			obj["Slope/Impact"]="Impact"
			if(ft.include_slope):obj["Slope/Impact"]="Slope"
			obj["Threshold"]="None"
			if(ft.threshold):obj["Threshold"]="Threshold"
			obj["Include Mean"]="No"
			if(ft.include_mean):obj["Include Mean"]="Yes"
			obj["ShareY"]="Local"
			if(ft.global_norm):obj["ShareY"]="Global"
			verbs.append(obj)
			short = False
	print(ft.toText())
pd.DataFrame(verbs).to_csv("../Boston_Housing/text_59.0.csv",index=False)

	
