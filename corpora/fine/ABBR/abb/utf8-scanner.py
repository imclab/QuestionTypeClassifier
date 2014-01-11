# -*- coding: utf-8 -*-
# import os
# path = os.path.dirname(__file__)
# print path

# for fname in path:
#     print fname
#     content = open(path + file).read()
#     for line in content:
#         try:
#             content.encode('ascii')
#         except UnicodeDecodeError:
#             print "%s contains non-ascii characters" % file

import os

for dirname, dirnames, filenames in os.walk('/home/gavin/dev/Factoid-Question-Answering/corpora/data/fine'):
    # print path to all filenames.
    for filename in filenames:
        fname = os.path.join(dirname, filename)
        print fname
        content = open(fname, 'r').read()
        print content
		for line in content:
			print line
	        try:
	            content.encode('ascii')
	        except UnicodeDecodeError:
	            print "%s contains non-ascii characters" % file