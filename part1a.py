from part1classifier import func

output = open('part1a.txt','w')
for i in range(1,5):
	for j in range(1,3):
		for k in range(1,3):
			func(i,j,output,k)
output.close()