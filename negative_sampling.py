import numpy as np

def main():
	max_line = 30835679
	num_postive = 355734
	num_negtive = num_postive*5

	np.random.seed(10)
	shuffle_indices = set(np.random.permutation(np.arange(max_line))[:num_negtive])

	index = 0
	f = open('../data/chinese-relation_extraction/pre_brcnn_data_train_n','r')
	h = open('../data/chinese-relation_extraction/pre_brcnn_data_train_p','r')
	g = open('../data/chinese-relation_extraction/pre_brcnn_data_train_negativesampling','w')
	line = f.readline()
	while line != '':
		if index in shuffle_indices:
			g.write(line)
		index += 1
		line = f.readline()
	line = h.readline()
	while line != '':
		g.write(line)
		line = h.readline()
	h.close()
	g.close()
	f.close()

	index = 0
	f = open('../data/chinese-relation_extraction/pre_brcnn_data_target_n','r')
	h = open('../data/chinese-relation_extraction/pre_brcnn_data_target_p','r')
	g = open('../data/chinese-relation_extraction/pre_brcnn_data_target_negativesampling','w')
	line = f.readline()
	while line != '':
		if index in shuffle_indices:
			g.write(line)
		index += 1
		line = f.readline()
	line = h.readline()
	while line != '':
		g.write(line)
		line = h.readline()
	h.close()
	g.close()
	f.close()

if __name__ == '__main__':
	main()