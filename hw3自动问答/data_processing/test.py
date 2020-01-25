import sklearn.metrics as mt 

bst_py_idx = []
bst_ry_idx = []

with open("answer/final_output_01_validation(1).txt",'r', encoding = "utf-8") as f:
	line = f.readline()
	while line:
		bst_py_idx.append(int(line[0]))
		line = f.readline()
with open("Data/data/validation-set.data", 'r', encoding = "utf-8") as f:
	line = f.readline()
	while line:
		text = line.strip("\n")
		split = text.split('\t', 2)
		if split[2] == "":
			print(split[1])
		bst_ry_idx.append(int(split[2]))
		line = f.readline()

assert len(bst_py_idx) == len(bst_ry_idx)
f = open("./score2.txt", 'w')
f1score1 = mt.f1_score( bst_py_idx, bst_ry_idx, average='macro' )
print( "macro-F1: ", f1score1 )
f.write("macro-F1: " + str(f1score1))
f.write('\n')
f1score2 = mt.f1_score( bst_py_idx, bst_ry_idx, average='micro' )
print( "micro-F1:", f1score2)
f.write("micro-F1: " + str(f1score2))
f.write('\n')

pscore1 = mt.precision_score(bst_py_idx, bst_ry_idx, average='macro')
print( "macro-P:", pscore1)
f.write("macro-P: " + str(pscore1))
f.write('\n')
pscore2 = mt.precision_score( bst_py_idx, bst_ry_idx, average='micro' )
print( "micro-P:", pscore2)
f.write("macro-P: " + str(pscore2))
f.write('\n')

rscore1 = mt.recall_score( bst_py_idx, bst_ry_idx, average='macro' )
print( "macro-R: ", rscore1 )
f.write("macro-R: " + str(rscore1))
f.write('\n')
rscore2 = mt.recall_score( bst_py_idx, bst_ry_idx, average='micro' )
print( "micro-R:", rscore2)
f.write("micro-R: " + str(rscore2))
f.write('\n')
f.close()