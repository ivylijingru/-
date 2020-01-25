import string
import json
import unicodedata

def load_prediction():
	res = json.load(open("Data/data/predictions_.json", 'r')) 
	i = 1
	result = []
	num = 0
	with open("Data/data/demo.txt", 'r', encoding = 'utf-8') as f:
		line = f.readline()
		while line:
			context = line[:-1]
			#context = context.replace(" ","")
			context = context.lower()
			#context = unicodedata.normalize('NFKD',context).encode('ascii', 'ignore').decode('utf-8')
			line = f.readline()
			temp1 = line.strip('\n')
			sentence_idx = temp1.split('\t')
			ans0 = res[str(i)]
			ans0 = ans0.replace(" ", "")
			ans = ""
			for i in range(len(ans0)):
				if ans0[i] < '\u4e00' or ans0[i] > '\u9fa5':
					ans += "?"
				else:
					ans += ans0[i]
			#print(context)
			#print(ans)

			idx0 = context.find(ans)
			idx1 = idx0 + len(ans)-1
			if(idx0 == -1):
				print(context)
				print(ans)
			#print(idx0)
			#print(idx1)
			flag = 0
			for j in range(len(sentence_idx)):
				if j == len(sentence_idx)-1:
					break
				#print(sentence_idx[j])
				if idx0 > int(sentence_idx[j]):
					result.append(0)
				elif flag == 0:
					result.append(0)
					flag = 1
				elif flag == 1:
					result.append(1)
					flag = 2
				else:
					result.append(0)
			if len(result) - num != len(sentence_idx) -1:
				print(i)
			line = f.readline()
			num = len(result)
			i += 1
			#print(str(i), ":", str(len(result)))
	with open("Data/data/result.txt", 'w', encoding = 'utf-8') as f:
		for i in range(len(result)):
			f.write(str(result[i]))
			f.write('\n')



if __name__=="__main__":
	load_prediction()