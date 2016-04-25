##############################################################################
#   Phonaesthemes network modeler
#   Mike Pham
##############################################################################

#   This is a set of scripts that will use a thesaurus to build semantic networks of related words.
#   This network is then restricted to subgraphs of words sharing similar (2-letter) prefixes.
#   Edges are bidirectional, and either unweighted (nodes are connected if they share each other as synonyms) or weighted using the Jaccard Index of their shared synonyms.
#   Word frequencies are provided by the COCA corpus
#   Graphs are in networkx format
#   Graphs outputed to gexf format for viewing with Gephi
#   Includes code to output JSON files for d3 implementation

##############################################################################



import random
import os
import subprocess
import numpy as np
import networkx as nx
import json
from networkx.algorithms import bipartite
import nltk
import collections
from NetworkxD3.NetworkxD3 import simpleNetworkx
from networkx.readwrite import json_graph
import community


filename = 'thesGraphSymLinks.txt'
file = open(filename,'r')
all_lines = file.read().splitlines()
file.close()

thesmap = {}

for line in all_lines:
    splitted = line.split(',')
    head = splitted[0]
    syns = splitted[1:]
    thesmap[head] = syns
    
###---------------------------------------------
### build word frequency dictionary from COCA corpus
###---------------------------------------------
corpusFile = open('coca_freqs_alpha.csv','r')

corpusAllLines = corpusFile.read().splitlines()
corpusFile.close()

cocaFreqDict = {}

for line in corpusAllLines:
    splitted = line.split(',')
    word = splitted[0]
    freq = int(splitted[1])
    cocaFreqDict[word] = freq

#--------------------------------------------------------------------------------
# Graph stuff - finding connected components 
#--------------------------------------------------------------------------------
# # Return roots and connected components of a graph represented as a dictionary 
# # whose keys are nodes and whose values are lists of nodes where key has val 
# # if there is an edge between key and val
#
# def getRoots(aNeigh):
#     def findRoot(aNode,aRoot):
#         while aNode != aRoot[aNode][0]:
#             aNode = aRoot[aNode][0]
#         return (aNode,aRoot[aNode][1])
#     myRoot = {} 
#     for myNode in aNeigh.keys():
#         myRoot[myNode] = (myNode,0)  
#     for myI in aNeigh: 
#         for myJ in aNeigh[myI]: 
#             (myRoot_myI,myDepthMyI) = findRoot(myI,myRoot) 
#             (myRoot_myJ,myDepthMyJ) = findRoot(myJ,myRoot) 
#             if myRoot_myI != myRoot_myJ: 
#                 myMin = myRoot_myI
#                 myMax = myRoot_myJ 
#                 if  myDepthMyI > myDepthMyJ: 
#                     myMin = myRoot_myJ
#                     myMax = myRoot_myI
#                 myRoot[myMax] = (myMax,max(myRoot[myMin][1]+1,myRoot[myMax][1]))
#                 myRoot[myMin] = (myRoot[myMax][0],-1) 
#     myToRet = {}
#     for myI in aNeigh: 
#         if myRoot[myI][0] == myI:
#             myToRet[myI] = []
#     for myI in aNeigh: 
#         myToRet[findRoot(myI,myRoot)[0]].append(myI) 
#     return myToRet 
# 
#------------------------------------------------------------------------------------------ 
# Stuff for computing clustering coefficients 
#------------------------------------------------------------------------------------------


def all_pairs(L):
    '''Returns all pairs of distinct elements from list L
       [1,2,3,4]->[(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]'''
    return [(L[i1],L[i2]) for i1 in range(len(L)) for i2 in range(i1+1,len(L))]

def nei_nei_frac(G,v):
    '''For node v in graph G, if AP is all pairs of neighbors of v
       return the fraction of AP that are neighbors of eachother.'''
    neighbors_of_v = G[v]
    neighbor_pairs = all_pairs(neighbors_of_v)
    neighbors_who_are_neighbors = [(x,y) for (x,y) in neighbor_pairs if y in G[x]]
    if len(neighbor_pairs) == 0:
        return 0
    return len(neighbors_who_are_neighbors)/float(len(neighbor_pairs))

def avg_clust_coef(G):
    '''Returns the average clustering coefficient of graph G 
       (i.e., avg of nei_nei_frac over all nodes of graph G)'''       
    nei_fracs = []
    for node in G.keys():
        f = nei_nei_frac(G,node)
        nei_fracs.append(f)
    return sum(nei_fracs)/len(nei_fracs)


# ------------------------------------------------------------------------------------
# Code for making subgraphs
# ------------------------------------------------------------------------------------
    
def wordsWithPrefix(pfx):
    '''returns a list of all words in the dictionary that begin with a given prefix'''
    return [wd for wd in thesmap if wd.startswith(pfx)]

def wordsWithSuffix(sfx):
	'''returns a list of all words in the dictionary that end with a suffix'''
	return [wd for wd in thesmap if wd.endswith(sfx)]

def wordsWithAffix(afx):
	'''returns a list of all words in the dictionary that contain an affix'''
	return [wd for wd in thesmap if afx in wd]

def NrandomWords(N):
    '''returns a list of N number of random words from the dictionary'''
    randomheads = random.sample(thesmap,N)
    return randomheads
    
def NrandomWordsFrom(pfx,N):
    '''returns a list of N number of random words beginning with a given prefix'''
    randomheads = random.sample(wordsWithPrefix(pfx),N)
    return randomheads

def subgraph(grph,nodelist):
    '''takes a graph represented as a dictionary from nodes to lists of nodes
       and returns a subgraph with only the nodes listed in nodelist'''
    result = {}
    for (key,vals) in grph.items():
        if key in nodelist:
            nuVals = [x for x in vals if x in nodelist]
            # trim empty lists here if we want to
            result[key] = nuVals
    return result 

def randomSubGraph(N):
    '''returns a subgraph for a list of N number of random words'''
    headlist = NrandomWords(N)
    return subgraph(thesmap,headlist)
    

def randomSubGraphFrom(pfxList,N):
    '''returns a subgraph for a list of N number of random words beginning
        with a given prefix'''
    headlist = NrandomWordsFrom(pfxList,N)
    return subgraph(thesmap,headlist)
            

def sampler(size,times):
    for x in range(times):
        G = randomSubGraph(size)
        cc1 = avg_clust_coef(G)
        print cc1


def avgavgcoef(size,times):
    r = []
    for x in range(times):

        G = randomSubGraph(size)
        cc1 = avg_clust_coef(G)
        r.append(cc1)
    return sum(r)/float(len(r))

def secondNeighbours(suburb,nhood):
    """returns list of neighbours of neighbours"""
    neighbourList = []
    for neighbour in nhood:
        nhood2 = suburb[neighbour]
        # print MrRogers,nhood,'   :   ',neighbour,nhood2
        # whoswho = ''
        for neighbour2 in nhood2:
            if neighbour2 in nhood:
                if (neighbour2,neighbour) not in neighbourList:
                    neighbourList += [(neighbour,neighbour2)]
                    # whoswho = whoswho + ', ' + neighbour2
        # print sharedNeighbours, whoswho, '\n'
    return neighbourList
    
def CCofNode(suburb,tupperwarePartyGuests):
    """returns clustering coefficient of a node given the first neighbours list, suburb,
    and the list of second neighbours, tupperwarePartyGuests"""
    maxPartyGuests = (len(suburb)*(len(suburb)-1))/2
    actualPartyGuests = len(tupperwarePartyGuests)
    
    if maxPartyGuests > 0:
        CC = float(actualPartyGuests)/float(maxPartyGuests)
    else:
        CC = 0
        
    return CC

def clusterCoef(somegrf):
    """returns the clustering coefficient of a graph somegrf"""
    MrRogersCoefficients = 0
    totalCC = 0
    for MrRogers in somegrf:
        nhood = somegrf[MrRogers]
        neighbourList = secondNeighbours(somegrf,nhood)
        MrRogersCoefficients = MrRogersCoefficients + CCofNode(nhood,neighbourList)
    
    totalCC = MrRogersCoefficients/len(somegrf)
    return totalCC
    
def randoClusterCoef(nhoodSize,iterations):
    """returns the list of clustering coefficients of random graphs of size nhoodSize
    based on iterations number of iterations """
    CCvalues = []
    for x in range(iterations):
        randoGrf = randomSubGraph(nhoodSize)
        CCvalues.append(clusterCoef(randoGrf))
    return CCvalues

def randoClusterCoefFrom(pfx,nhoodSize,iterations):
    """returns the list of clustering coefficients of random graphs of size nhoodSize
    based on iterations number of iterations """
    CCvalues = []
    for x in range(iterations):
        randoGrf = randomSubGraphFrom(pfx,nhoodSize)
        CCvalues.append(clusterCoef(randoGrf))
    return CCvalues
    
def Jaccard(lst1,lst2):
    '''returns the Jaccard index: similarity measure of two sets defined by number of
    shared items (A meets B) over total number of items (A join B)'''
    # s1 = set(lst1.split(','))
    # s2 = set(lst2.split(','))
    s1 = set(lst1)
    s2 = set(lst2)
    stotal = len(s1|s2)
    sint = float(len(s1&s2))    
    Jindex = round(sint/stotal, 6)
    return Jindex


def meanDegree(G,SG):
    '''returns the mean degree of all nodes in a subgraph'''
    Degrees = []
    for Word in SG:
        if SG[Word]:
            Degrees.append(G.degree(Word))
        else:
            Degrees.append(0)
        # print pfxDegrees
    return np.mean(Degrees)

def stdDegree(G,SG):
    '''returns the standard deviation of the degrees of all nodes in a subgraph'''
    Degrees = []
    for Word in SG:
        if SG[Word]:
            Degrees.append(G.degree(Word))
        else:
            Degrees.append(0)
        # print pfxDegrees
    return np.std(Degrees)

def meanCCnx(G):
    '''returns the mean clustering coefficient in a subgraph'''
    nodeCCs = []
    for CC in nx.clustering(G):
        # print CC
        nodeCCs.append(nx.clustering(G,CC))
    return np.mean(nodeCCs)

def stdCCnx(G):
    '''returns the standard deviation of clustering coefficients in a subgraph'''
    nodeCCs = []
    for CC in nx.clustering(G):
        # print CC
        nodeCCs.append(nx.clustering(G,CC))
    return np.std(nodeCCs)

def cocaFreqLog(word):
    '''returns the log10 frequency of a given word in the corpus (COCA)'''
    wordFreq = cocaFreqDict[word]
    return float(np.log10(wordFreq))

# ------------------------------------------------------------------------------------
# Code for subgraphs for various prefixes
# ------------------------------------------------------------------------------------
    
def graphme(moniker,somegrf):
    outlinksname = 'graph_'+moniker+'.dot'
    os.chdir('graphs')
    outfile = open(outlinksname,'w')
    
    outfile.write('graph G {\n')

    links = []
    for (head,valset) in somegrf.items():
        outfile.write('    '+head+';\n')
        # print head
        
        for V in valset:
            if V != head:
                if (V,head) not in links:
                    links += [(head,V)]
                    outlinksString = '    '+head+' -- '+V+';\n'
                    outfile.write(outlinksString)
                    # print outlinksString    
    
    outfile.write('}')
    outfile.close()
    os.chdir('..')            
                
    # print '\n\n'
    # for (x,y) in links:
    #     print x,'<-->',y    
        
    # graphmerandom(pfx)    

# graphme('rando466',randomSubGraph(466))
# print len(wordsWithPrefix('st'))

# ###----------------------------------------------
# ### write .gext files for 'tru-', 'dru-'; ;'-mpf'/'pf'
# ###----------------------------------------------
# trumpPfxs = ['tru','dru']
# trumpSfxs = ['mp','mpf']
# trumpAfxs = ['pf']

# trumpOutfile = open('trumpDrumpf.csv','w')
# trumpOutfile.write('prefix,graph size,mean degree,degree standard dev,meanCC,CC standard dev\n')

# os.chdir('trump_graphs')

# for pfx in trumpPfxs:
# 	wordsWithPFX = wordsWithPrefix(pfx)
# 	print pfx, str(wordsWithPFX)

# 	gexfFileName = pfx + '_subgraph_jaccard_coca.gexf'
# 	gexfFile = open(gexfFileName, 'w')

# 	G = nx.Graph()
# 	degreeList = []

# 	for word1 in wordsWithPFX:
# 		degree = 0
#         wordsWithPFX.remove(word1)
#         for word2 in wordsWithPFX:
#             edgeWeight = Jaccard(thesmap[word1],thesmap[word2])
#             if edgeWeight > 0:
#                 degree = degree + 1
#                 G.add_edge(word1, word2, weight=edgeWeight)
#             else:
#             	G.add_node(word1)
#             	G.add_node(word2)

#                 ## for COCA corpus frequencies ###
#                 if word1 in cocaFreqDict:
#                     G.node[word1]['cocaFreq [z]'] = cocaFreqLog(word1)
#                 else:
#                     G.node[word1]['cocaFreq [z]'] = 0

#                 print "{}\t{}\t{}".format(word1,word2,str(edgeWeight))
#                 # print str(G.node[word1]), str(G.node[word2]), str(edgeWeight)
#         degreeList.append(degree)

# 	nx.write_gexf(G, gexfFileName)
# 	gexfFile.close()

# 	trumpOutfile.write(pfx + ',' + str(len(G)) + ',' + str(np.mean(degreeList)) + ',' + str(np.std(degreeList)) + ',' + str(meanCCnx(G)) + ',' + str(stdCCnx(G)) + '\n')

# for sfx in trumpSfxs:
# 	wordsWithSFX = wordsWithSuffix(sfx)
# 	print sfx, str(wordsWithSFX)

# 	gexfFileName = sfx + 'sfx_subgraph_jaccard_coca.gexf'
# 	gexfFile = open(gexfFileName, 'w')

# 	G = nx.Graph()
# 	degreeList = []

# 	if wordsWithSFX:
# 		for word1 in wordsWithSFX:
# 			degree = 0
# 	        wordsWithSFX.remove(word1)
# 	        for word2 in wordsWithSFX:
# 	            edgeWeight = Jaccard(thesmap[word1],thesmap[word2])
# 	            if edgeWeight > 0:
# 	                degree = degree + 1
# 	                G.add_edge(word1, word2, weight=edgeWeight)
# 	            else:
# 	            	G.add_node(word1)
# 	            	G.add_node(word2)

# 	                ## for COCA corpus frequencies ###
# 	                if word1 in cocaFreqDict:
# 	                    G.node[word1]['cocaFreq [z]'] = cocaFreqLog(word1)
# 	                else:
# 	                    G.node[word1]['cocaFreq [z]'] = 0

# 	                print "{}\t{}\t{}".format(word1,word2,str(edgeWeight))
# 	                # print str(G.node[word1]), str(G.node[word2]), str(edgeWeight)
# 	        degreeList.append(degree)

# 		nx.write_gexf(G, gexfFileName)
# 		gexfFile.close()

# 		trumpOutfile.write(pfx + ',' + str(len(G)) + ',' + str(np.mean(degreeList)) + ',' + str(np.std(degreeList)) + ',' + str(meanCCnx(G)) + ',' + str(stdCCnx(G)) + '\n')

# for afx in trumpAfxs:
# 	wordsWithAFX = wordsWithAffix(afx)
# 	print afx, str(wordsWithAFX)

# 	gexfFileName = afx + '_afx_subgraph_jaccard_coca.gexf'
# 	gexfFile = open(gexfFileName, 'w')

# 	G = nx.Graph()
# 	degreeList = []

# 	if wordsWithAFX:
# 		for word1 in wordsWithAFX:
# 			degree = 0
# 	        wordsWithAFX.remove(word1)
# 	        for word2 in wordsWithAFX:
# 	            edgeWeight = Jaccard(thesmap[word1],thesmap[word2])
# 	            if edgeWeight > 0:
# 	                degree = degree + 1
# 	                G.add_edge(word1, word2, weight=edgeWeight)
# 	            else:
# 	            	G.add_node(word1)
# 	            	G.add_node(word2)

# 	                ## for COCA corpus frequencies ###
# 	                if word1 in cocaFreqDict:
# 	                    G.node[word1]['cocaFreq [z]'] = cocaFreqLog(word1)
# 	                else:
# 	                    G.node[word1]['cocaFreq [z]'] = 0

# 	                print "{}\t{}\t{}".format(word1,word2,str(edgeWeight))
# 	                # print str(G.node[word1]), str(G.node[word2]), str(edgeWeight)
# 	        degreeList.append(degree)

# 		nx.write_gexf(G, gexfFileName)
# 		gexfFile.close()

# 		trumpOutfile.write(pfx + ',' + str(len(G)) + ',' + str(np.mean(degreeList)) + ',' + str(np.std(degreeList)) + ',' + str(meanCCnx(G)) + ',' + str(stdCCnx(G)) + '\n')



# os.chdir('..')
# trumpOutfile.close()

###---------------------------------------------------------
###     export networkx graphs as json for d3 visualization
###---------------------------------------------------------
alphabet = 'abcdefghijklmnopqrstuvwxyz'
# alphabet = 'abc'          #  small test alphabet

os.chdir('d3-graphs/json_files')

twoLtrPfxs = []
for a in alphabet:
    for b in alphabet:
        twoLtrPfx = a + b
        twoLtrPfxs.append(twoLtrPfx)

print twoLtrPfxs

for pfx in twoLtrPfxs:
    # if pfx != 'ko' and pfx != 'ow':
    wordsWithPFX = wordsWithPrefix(pfx)
    print pfx, str(wordsWithPFX)

    if len(wordsWithPFX) > 2:
        pfxFileName = pfx + '_jaccard.json'
        outfile1 = open(pfxFileName, 'w')

        G = nx.Graph()

        for word1 in wordsWithPFX:
            print word1
            G.add_node(word1)
            degree = 0
            wordsWithPFX.remove(word1)

            for word2 in wordsWithPFX:
                edgeWeight = Jaccard(thesmap[word1],thesmap[word2])
                if edgeWeight > 0:
                    degree = degree + 1
                    G.add_edge(word1, word2, weight=edgeWeight)
                    print word1 + '--' + word2

            G.node[word1]['degree'] = degree

        if nx.average_clustering(G) > 0:
            part = community.best_partition(G)

            for word in part:
                G.node[word]['group'] = part[word]

        else:
            groupCount = 0
            for word in G:
                G.node[word]['group'] = groupCount
                groupCount =+ 1

        outfile1.write(json.dumps(json_graph.node_link_data(G)))
        outfile1.close()

###-----------------------------------------
###  write .gexf files for each 2 letter prefix
###  !!edges determined by Jaccard index INSTEAD of shared synonyms
###-----------------------------------------
# alphabet = 'abcdefghijklmnopqrstuvwxyz'
# # alphabet = 'abc'          #  small test alphabet

# '''rewrite all subgraph files as .gexf files for Gephi
#     !!edges determined by Jaccard index of two nodes (=weight)
#     This is instead of determining an edge by whether two words share
#     each other as synonyms'''

# infoFile = open('allPfx_graph_info_coca.csv', 'w')
# infoFile.write('prefix,graph size,mean degree,degree standard dev,meanCC,CC standard dev\n')

# # os.chdir('gexf_graphs') #use for unweighted edges
# # os.chdir('gexf_graphs_jaccard')    #use for weighted edges
# # os.chdir('gexf_graphs_jaccard_brown')   #use for weighted edges and node attribute = word frequency from Brown corpus
# os.chdir('gexf_graphs_jaccard_coca')    #use for weighted edges and node attribute = word frequency from COCA corpus


# for a in alphabet:
#     for b in alphabet:
#         pfx = a+b
#         wordsWithPFX = wordsWithPrefix(pfx)
#         print pfx, str(wordsWithPFX)

#         if len(wordsWithPFX) > 2:
#             # gexfFileName = pfx + '_subgraph_jaccard.gexf'   #for jaccard index weighted edges
#             # gexfFileName = pfx + '_subgraph_jaccard_brown.gexf' #for jaccard index weighte edges and brown corpus frequency node attribute
#             gexfFileName = pfx + '_subgraph_jaccard_coca.gexf'  #for jaccard index weighted edges and COCA corpus frequency node attribute
#             gexfFile = open(gexfFileName, 'w')

#             G = nx.Graph()

#             degreeList = []

#             for word1 in wordsWithPFX:
#                 degree = 0
#                 wordsWithPFX.remove(word1)
#                 for word2 in wordsWithPFX:
#                     edgeWeight = Jaccard(thesmap[word1],thesmap[word2])
#                     if edgeWeight > 0:
#                         degree = degree + 1
#                         G.add_edge(word1, word2, weight=edgeWeight)

#                         ## for Brown corpus frequencies ###
#                         # if brownFreqLog(word1):
#                         #     G.node[word1]['brownFreq [z]'] = brownFreqLog(word1)    #add brown corpus frequency
#                         # else:
#                         #     G.node[word1]['brownFreq [Z]'] = 0

#                         ## for COCA corpus frequencies ###
#                         if word1 in cocaFreqDict:
#                             G.node[word1]['cocaFreq [z]'] = cocaFreqLog(word1)
#                         else:
#                             G.node[word1]['cocaFreq [z]'] = 0

#                         print "{}\t{}\t{}".format(word1,word2,str(edgeWeight))
#                         # print str(G.node[word1]), str(G.node[word2]), str(edgeWeight)
#                 degreeList.append(degree)

#             nx.write_gexf(G, gexfFileName)
#             gexfFile.close()

#             infoFile.write(pfx + ',' + str(len(G)) + ',' + str(np.mean(degreeList)) + ',' + str(np.std(degreeList)) + ',' + str(meanCCnx(G)) + ',' + str(stdCCnx(G)) + '\n')

# os.chdir('..')
# infoFile.close()


##-----------------------------------------
##  write .gexf files for each 2 letter prefix
##-----------------------------------------
# alphabet = 'abcdefghijklmnopqrstuvwxyz'
# # alphabet = 'abc'          #  small test alphabet
#
# '''rewrite all subgraph files as .gexf files for Gephi
#     option to include Jaccard index as edge weight between 2 nodes'''
#
# infoFile = open('allPfx_graph_info.csv', 'w')
# infoFile.write('prefix,graph size,mean degree,degree standard dev,mean CC,CC standard dev\n')
#
# # os.chdir('gexf_graphs') #use for unweighted edges
# os.chdir('gexf_graphs_weighted')    #use for weighted edges
#
# for a in alphabet:
#     for b in alphabet:
#         pfx = a+b
#         if subgraph(thesmap,wordsWithPrefix(pfx)):
#             # gexfFileName = pfx + '_subgraph.gexf'   #use for unweighted edges
#             gexfFileName = pfx + '_subgraph_weighted.gexf'   #use for weighted edges
#             gexfFile = open(gexfFileName, 'w')
#
#             G = nx.Graph()
#
#             pfxSG = subgraph(thesmap, wordsWithPrefix(pfx))
#             for head in pfxSG:
#                 for syn in pfxSG[head]:
#                     edgeWeight = Jaccard(thesmap[head],thesmap[syn])
#                     # G.add_edge(head, syn) #use for unweighted edges
#                     G.add_edge(head, syn, weight=edgeWeight)    #use for weighted edges
#                     print head, syn
#
#             nx.write_gexf(G, gexfFileName)
#             gexfFile.close()
#
#             infoFile.write(pfx + ',' + str(len(G)) + ',' + str(meanDegree(G,pfxSG)) + ',' + str(stdDegree(G,pfxSG)) + ',' + str(meanCCnx(G)) + ',' + str(stdCCnx(G)) + '\n')
#
#
# os.chdir('..')
# infoFile.close()

###---------------------------------------------------------
###     subgraph gexf files for iteratively built prefix
###---------------------------------------------------------
# fullPfx = raw_input('real prefix (>2 symbols): ')
# directoryString = 'gexf_graphs/' + fullPfx
# metricsFilename = fullPfx + '_subgraph_metrics.csv'
#
# os.mkdir(directoryString)
# os.chdir(directoryString)
# outfileMetrics = open(metricsFilename, 'w')
# outfileMetrics.write('prefix,graph size, mean degree, degree standard dev, mean CC, CC standard dev, # connected components\n')
#
#
# pfx = ''
# for segment in fullPfx:
#     pfx = pfx + segment
#
#     if subgraph(thesmap,wordsWithPrefix(pfx)):
#         outfilename = pfx + '_subgraph.gexf'
#         outfile = open(outfilename, 'w')
#
#         G = nx.Graph()
#
#         pfxSG = subgraph(thesmap, wordsWithPrefix(pfx))
#         for head in pfxSG:
#             for syn in pfxSG[head]:
#                 G.add_edge(head, syn)
#                 print head, syn
#
#         nx.write_gexf(G, outfilename)
#
#         graphSize = str(len(G))
#         connectComponents = str(nx.number_connected_components(G))
#
#         metrics = pfx + ',' + graphSize + ',' + str(meanDegree(G,pfxSG)) + ',' + str(stdDegree(G,pfxSG)) + ',' + str(meanCCnx(G)) + ',' + str(stdCCnx(G))+ ',' + connectComponents + '\n'
#
#         outfileMetrics.write(metrics)
#
#         print metrics
#
#
# os.chdir('../')
# outfileMetrics.close()
# outfile.close()


###--------------------------------------------
###     network metrics for prefix subgraphs
###--------------------------------------------
# for a in alphabet:
#     for b in alphabet:
#         pfx = a+b
#         if subgraph(thesmap,wordsWithPrefix(pfx)):
#             SG = subgraph(thesmap,wordsWithPrefix(pfx))
#             G = nx.Graph()
#
#             pfxSG = subgraph(thesmap, wordsWithPrefix(pfx))
#             for head in pfxSG:
#                 for syn in pfxSG[head]:
#                     G.add_edge(head, syn)
#
#             # print G.nodes()
#             # print G.edges()
#
#             '''put degree of all nodes in pfx subgraph into list'''
#             pfxDegrees = []
#             for pfxWord in pfxSG:
#                 if pfxSG[pfxWord]:
#                     pfxDegrees.append(G.degree(pfxWord))
#                 else:
#                     pfxDegrees.append(0)
#
#             '''calculate:
#              1. graph size
#              2. mean degree of all nodes in pfx subgraph
#              3. average clustering coefficient'''
#             graphSize = str(len(G))
#             meanDegree = str(np.mean(pfxDegrees))
#
#             nodeCCs = []
#             for CC in nx.clustering(G):
#                 nodeCCs.append(nx.clustering(G,CC))
#
#             print nodeCCs
#             meanCC = str(np.mean(nodeCCs))
#
#             outfile.write(pfx + ',' + graphSize + ',' + meanDegree + ',' + meanCC + '\n')
#
#             print pfx + ' (size, avgDeg, avgCC): ' + graphSize + ',' + meanDegree + ',' + meanCC
#
# outfile.close()