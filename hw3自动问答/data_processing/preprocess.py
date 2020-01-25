import string
import io
#import pkuseg

ques = []
context = []
ans_idx = []
ans = []
sentence = []

def load_data():
	with open( "Data/data/test-set.data", "r", encoding = 'utf-8') as f0:
		line = f0.readline()
		ques_prev = ""
		ans_num = 0
		context_all = ""
		ans_all = []
		ans_idx_all = []
		total = 0
		sentence_idx = []

		while line:
			text = line[:-1]
			#print(text)
			#print(ques_prev)


			split = text.split('\t', 2)
			ques_now = split[0]
			ans_now = split[1]
			label = split[2]

			#ans_now = ans_now.replace(" ","")

			#print(ques_now)

			if ques_now == ques_prev:
				#print(1)
				#print(label)
				if label == "0":
					context_all += ans_now
					#print(context_all)
					#ans_tokens = seg.cut(ans_now)
					#ans_seq = ""
					#for w in range(len(ans_tokens)):
					#	ans_seq += ans_tokens[w]
					#	ans_seq += " "
					#context_all += ans_seq
				elif label == "1":
					ans_num = len(context_all)
					ans_num += 1
					ans_idx_all.append(str(ans_num))
					#ans_tokens = seg.cut(ans_now)
					#ans_seq = ""
					#for w in range(len(ans_tokens)):
					#	ans_seq += ans_tokens[w]
					#	ans_seq += " "
					#context_all += ans_seq
					#ans_all = ans_seq
					context_all += ans_now
					#print(context_all)
					ans_all.append(ans_now)
					#print(str(label))
					#print(ans_all)
				sentence_idx.append(len(context_all))
			else:
				with open("Data/data/demo_test.txt", "a+", encoding = "utf-8") as f:
					total += 1
					#f.write(str(total))
					#f.write("context:\n")
					f.write(context_all)
					f.write("\n")
					for i in range(len(sentence_idx)):
						f.write(str(sentence_idx[i]))
						f.write("\t")
					#f.write("question:\n")
					#f.write(ques_prev)
					#f.write("\n")
					#f.write("answer:\n")
					#f.write(str(ans_all))
					#f.write("\n")
					#f.write(str(ans_idx_all))
					f.write("\n")
				
				if ans_all != []:
					context.append(context_all)
					#ques_tokens = seg.cut(ques_prev)
					#ques_seq = ""
					#for w in range(len(ques_tokens)):
					#	ques_seq += ques_tokens[w]
					#	ques_seq += " "
					#ques.append(ques_seq)
					ques.append(ques_prev)
					ans_idx.append(ans_idx_all)
					ans.append(ans_all)
					sentence.append(sentence_idx)
				else:
					context.append(context_all)
					ques.append(ques_prev)
					ans_idx.append([0])
					ans.append([""])
					sentence_idx.append(sentence_idx)
					print(total)
				ques_prev = ques_now
				ans_num = 0
				ans_idx_all = []
				ans_all = []
				sentence_idx = []
				context_all = ans_now
				sentence_idx.append(len(context_all))
				if label == "1":
					ans_all.append(ans_now)
					ans_idx_all.append("0")
				

			line = f0.readline()

		with open("Data/data/demo_test.txt", "a+", encoding = "utf-8") as f:
			total += 1
			#f.write(str(total))
			#f.write("context:\n")
			f.write(context_all)
			f.write("\n")
			for i in range(len(sentence_idx)):
				f.write(str(sentence_idx[i]))
				f.write("\t")
			#f.write("question:\n")
			#f.write(ques_prev)
			#f.write("\n")
			#f.write("answer:\n")
			#f.write(str(ans_all))
			#f.write("\n")
			#f.write(str(ans_idx_all))
			f.write("\n")

		if ans_all != []:
			context.append(context_all)
			#ques_tokens = seg.cut(ques_prev)
			#ques_seq = ""
			#for w in range(len(ques_tokens)):
			#	ques_seq += ques_tokens[w]
			#	ques_seq += " "
			#ques.append(ques_seq)
			ques.append(ques_prev)
			ans_idx.append(ans_idx_all)
			ans.append(ans_all)
			sentence.append(sentence_idx)
		else:
			context.append(context_all)
			ques.append(ques_prev)
			ans_idx.append([0])
			ans.append([""])
			sentence_idx.append(sentence_idx)
			print(total)

	return

def write_json():
	with open("Data/data/test-set.json", "w", encoding="utf-8") as f:
		f.write("{\"data\":[")
		count = 0
		for i in range(len(context)):
			if i == 0:
				continue
			f.write("{\"title\":\"none\",\"paragraphs\":[{\"context\":\"")
			f.write(context[i].replace("\\", "\\\\").replace("\n", "\\\n").replace("\r", "\\\r").replace("\t", "\\\t").replace("\"","\\\""))
			f.write("\",\"qas\":[")
			for j in range(len(ans_idx[i])):
				f.write("{\"answers\":[{\"answer_start\":")
				f.write(str(ans_idx[i][j]))
				f.write(",\"text\":\"")
				f.write(ans[i][j].replace("\\", "\\\\").replace("\n", "\\\n").replace("\r", "\\\r").replace("\t", "\\\t").replace("\"","\\\""))
				f.write("\"}],\"question\":\"")
				f.write(ques[i].replace("\\", "\\\\").replace("\n", "\\\n").replace("\r", "\\\r").replace("\t", "\\\t").replace("\"","\\\""))
				f.write("\",\"id\":\"")
				f.write(str(i))
				if j == len(ans_idx[i]) - 1:
					f.write("\"}]}]}")
				else:
					f.write("\"},")
			if i != len(context)-1:
				f.write(",")
			else:
				f.write("],")
		f.write("\"version\":\"v1.1\"}")
	return

if __name__ == "__main__":
	load_data()
	#write_json()
